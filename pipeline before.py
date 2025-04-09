import yaml
import queue
import threading
import time
import prometheus_client
from typing import Optional, Dict, Any
from components.socket_server import ARSocketServer
from components.detection import ConduitDetector
from components.tracking import ConduitTracker
from components.data_manage import ARDatabase

# Prometheus监控指标
FRAME_COUNTER = prometheus_client.Counter(
    'ar_pipeline_frames_total', 
    'Total processed frames'
)
ERROR_COUNTER = prometheus_client.Counter(
    'ar_pipeline_errors_total',
    'Total processing errors',
    ['stage']
)
LATENCY = prometheus_client.Histogram(
    'ar_pipeline_latency_seconds',
    'Processing latency per stage',
    ['stage'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5]
)

class Processor:
    """Pipeline处理器基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.input_queue = queue.Queue(maxsize=100)
        self.output_queue = queue.Queue(maxsize=100)
        self._stop_event = threading.Event()
        
    def process(self, data: Any) -> Any:
        """处理数据并返回结果"""
        raise NotImplementedError
        
    def run(self):
        """运行处理线程"""
        while not self._stop_event.is_set():
            try:
                data = self.input_queue.get(timeout=1)
                if data is None:  # 终止信号
                    break
                    
                result = self.process(data)
                if result is not None:
                    self.output_queue.put(result)
                    
            except queue.Empty:
                continue
            except Exception as e:
                ERROR_COUNTER.labels(self.__class__.__name__).inc()
                print(f"Error in {self.__class__.__name__}: {str(e)}")
                
    def stop(self):
        """停止处理器"""
        self._stop_event.set()
        self.input_queue.put(None)  # 发送终止信号

class ImageDecoder(Processor):
    """图像解码处理器"""
    
    def process(self, socket_data: bytes) -> Optional[Dict]:
        with LATENCY.labels('decode').time():
            try:
                # 提取帧数据 (跳过4字节长度头)
                frame_data = socket_data[4:-4]  # 最后4字节是CRC
                
                # CRC校验
                crc = struct.unpack('!I', socket_data[-4:])[0]
                if zlib.crc32(frame_data) != crc:
                    raise ValueError("CRC mismatch")
                    
                return {'raw_frame': frame_data}
                
            except Exception as e:
                ERROR_COUNTER.labels('decode').inc()
                return {'error': 'decode_failed', 'request_retry': True}

class Detector(Processor):
    """YOLO检测处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.detector = ConduitDetector(
            model_path=config.get('model_path', 'yolov8n.pt')
        )
        
    def process(self, data: Dict) -> Optional[Dict]:
        if 'error' in data:
            return data
            
        with LATENCY.labels('detection').time():
            try:
                detections = self.detector.detect(data['raw_frame'])
                data['detections'] = detections
                return data
            except Exception as e:
                ERROR_COUNTER.labels('detection').inc()
                return {'error': 'detection_failed'}

class Tracker(Processor):
    """DeepSORT跟踪处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        detector = ConduitDetector()
        self.tracker = ConduitTracker(detector)
        
    def process(self, data: Dict) -> Optional[Dict]:
        if 'error' in data:
            return data
            
        with LATENCY.labels('tracking').time():
            try:
                # 解码帧图像用于跟踪
                frame = cv2.imdecode(
                    np.frombuffer(data['raw_frame'], np.uint8), 
                    cv2.IMREAD_COLOR
                )
                
                # 执行跟踪
                results, _ = self.tracker.update(frame)
                data['tracks'] = results
                return data
            except Exception as e:
                ERROR_COUNTER.labels('tracking').inc()
                return {'error': 'tracking_failed'}

class DataSaver(Processor):
    """数据存储处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.db = ARDatabase(config.get('db_path', ':memory:'))
        self.retry_count = config.get('retry_count', 3)
        
    def process(self, data: Dict) -> Optional[Dict]:
        if 'error' in data:
            return data
            
        with LATENCY.labels('storage').time():
            attempts = 0
            while attempts < self.retry_count:
                try:
                    # 保存跟踪结果到数据库
                    for track in data['tracks']:
                        self.db.add_tracking({
                            'object_id': track['track_id'],
                            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                            'camera_coords': track['bbox'],
                            'world_coords': [0, 0, 0]  # 实际应用中从其他传感器获取
                        })
                    return data
                except Exception as e:
                    attempts += 1
                    if attempts == self.retry_count:
                        ERROR_COUNTER.labels('storage').inc()
                        return {'error': 'storage_failed'}

class ResponseBuilder(Processor):
    """响应构建处理器"""
    
    def process(self, data: Dict) -> Optional[Dict]:
        if 'error' in data:
            return {
                'status': 'error',
                'error_type': data['error'],
                'request_retry': data.get('request_retry', False)
            }
            
        with LATENCY.labels('response').time():
            return {
                'status': 'success',
                'tracks': data['tracks'],
                'timestamp': time.time()
            }

class ARPipeline:
    """AR装配处理流水线"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        # 加载配置
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        # 初始化处理器
        self.processors = [
            ImageDecoder(self.config),
            Detector(self.config),
            Tracker(self.config),
            DataSaver(self.config),
            ResponseBuilder(self.config)
        ]
        
        # 连接处理器队列
        for i in range(len(self.processors)-1):
            self.processors[i].output_queue = self.processors[i+1].input_queue
            
        # 启动Prometheus监控
        prometheus_client.start_http_server(
            self.config.get('metrics_port', 8000)
        )
        
    def start(self):
        """启动处理流水线"""
        for processor in self.processors:
            threading.Thread(
                target=processor.run,
                daemon=True
            ).start()
            
    def process_frame(self, frame_data: bytes) -> Dict:
        """处理单帧数据"""
        self.processors[0].input_queue.put(frame_data)
        return self.processors[-1].output_queue.get()
        
    def stop(self):
        """停止流水线"""
        for processor in self.processors:
            processor.stop()

# 示例配置 (config.yaml)
DEFAULT_CONFIG = """
socket:
  host: 0.0.0.0
  port: 8888
  
model:
  path: yolov8n.pt
  conf_thresh: 0.7
  nms_thresh: 0.5
  
tracking:
  max_age: 30
  min_hits: 3
  
database:
  path: ar_data.db
  
metrics:
  port: 8000
"""

if __name__ == '__main__':
    # 初始化并启动流水线
    pipeline = ARPipeline()
    pipeline.start()
    
    # 启动Socket服务器
    server = ARSocketServer()
    server.start()
