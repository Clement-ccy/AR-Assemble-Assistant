import cv2
import torch
import numpy as np
from pathlib import Path
from threading import Thread
from time import sleep
from ultralytics import YOLO
from typing import List, Dict
import json

class ConduitDetector:
    """导管专用YOLOv8检测器"""
    
    def __init__(self, model_path: str = 'yolov8n.pt'):
        self.model_path = model_path
        self.model = self._load_model()
        self.class_id = None  # conduit类别ID
        self.conf_thresh = 0.7
        self.nms_thresh = 0.5
        self._init_class_id()
        self.warmup()
        
        # 热更新线程
        self._stop_flag = False
        self.watcher = Thread(target=self._watch_model_update, daemon=True)
        self.watcher.start()
    
    def _load_model(self) -> YOLO:
        """加载模型并转移到GPU"""
        model = YOLO(self.model_path)
        if torch.cuda.is_available():
            model.to('cuda')
        return model
    
    def _init_class_id(self):
        """初始化导管类别ID"""
        # 处理不同版本的names属性格式
        names = getattr(self.model, 'names', {})
        
        if isinstance(names, dict):
            # 字典格式 {id: name}
            for idx, name in names.items():
                if isinstance(name, str) and 'conduit' in name.lower():
                    self.class_id = idx
                    break
        elif isinstance(names, (list, tuple)):
            # 列表格式 [name1, name2,...]
            for idx, name in enumerate(names):
                if isinstance(name, str) and 'conduit' in name.lower():
                    self.class_id = idx
                    break
        
        if self.class_id is None:
            self.class_id = 0  # 默认使用第一个类别
            print(f"Warning: 'conduit' class not found, using default class_id={self.class_id}")
    
    def warmup(self, img_size: int = 640):
        """预热模型"""
        dummy = torch.zeros((1, 3, img_size, img_size), device=self.model.device)
        self.model(dummy)
    
    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """图像预处理"""
        # OpenCV解码
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # 直方图均衡化 (YUV空间)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        return img
    
    def detect(self, image_bytes: bytes) -> List[Dict]:
        """执行检测"""
        img = self.preprocess(image_bytes)
        
        # 推理
        with torch.no_grad():
            results = self.model(img, verbose=False)[0]
        
        # 过滤结果
        boxes = []
        for box in results.boxes:
            if box.cls == self.class_id and box.conf >= self.conf_thresh:
                boxes.append({
                    'bbox': box.xywh[0].tolist(),  # [x,y,w,h]格式
                    'conf': float(box.conf)
                })
        
        # NMS处理
        if len(boxes) > 1:
            boxes = self._apply_nms(boxes)
            
        return boxes
    
    def _apply_nms(self, boxes: List[Dict]) -> List[Dict]:
        """非极大值抑制"""
        boxes_array = np.array([b['bbox'] for b in boxes])
        confs = np.array([b['conf'] for b in boxes])
        
        # 转换xywh为xyxy
        xyxy = boxes_array.copy()
        xyxy[:, 0] -= xyxy[:, 2] / 2  # x1 = x - w/2
        xyxy[:, 1] -= xyxy[:, 3] / 2  # y1 = y - h/2 
        xyxy[:, 2] += xyxy[:, 0]      # x2 = x1 + w
        xyxy[:, 3] += xyxy[:, 1]      # y2 = y1 + h
        
        # 执行NMS
        indices = cv2.dnn.NMSBoxes(
            xyxy.tolist(),
            confs.tolist(),
            self.conf_thresh,
            self.nms_thresh
        )
        
        return [boxes[i] for i in indices.flatten()]
    
    def _watch_model_update(self):
        """监控模型文件变化实现热更新"""
        last_mtime = Path(self.model_path).stat().st_mtime
        while not self._stop_flag:
            sleep(5)  # 每5秒检查一次
            current_mtime = Path(self.model_path).stat().st_mtime
            if current_mtime > last_mtime:
                print("Detected model update, reloading...")
                self.model = self._load_model()
                last_mtime = current_mtime
    
    def get_gpu_optimization_tips(self) -> Dict:
        """获取GPU显存优化建议"""
        tips = {
            'current_memory': f"{torch.cuda.memory_allocated()/1024**2:.2f}MB",
            'recommendations': [
                "使用半精度(fp16)推理: model.to(torch.float16)",
                "启用torch.backends.cudnn.benchmark=True",
                "减少输入分辨率(如640->320)",
                "使用torch.cuda.empty_cache()定期清理缓存",
                "批处理推理(多帧同时处理)"
            ]
        }
        if torch.cuda.is_available():
            tips['device'] = torch.cuda.get_device_name(0)
            tips['total_memory'] = f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB"
        return tips
    
    def __del__(self):
        self._stop_flag = True
        if hasattr(self, 'watcher') and self.watcher.is_alive():
            self.watcher.join(timeout=1)

# 示例用法
if __name__ == '__main__':
    detector = ConduitDetector()
    with open('test.jpg', 'rb') as f:
        img_bytes = f.read()
    
    results = detector.detect(img_bytes)
    print("Detection Results:", results)
    
    print("\nGPU Optimization Tips:")
    print(json.dumps(detector.get_gpu_optimization_tips(), indent=2))
