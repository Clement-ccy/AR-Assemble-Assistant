import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import List, Dict, Tuple
from components.detection import ConduitDetector
import cv2

class ConduitTracker:
    """导管连续跟踪器(DeepSORT实现)"""
    
    def __init__(self, detector: ConduitDetector):
        self.detector = detector
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_cosine_distance=0.2,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
            
        )
        self.track_history = {}  # 轨迹历史 {track_id: [points]}
        self.next_frame = 1
        
    def update(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """更新跟踪状态并返回结果"""
        # 检测导管
        _, jpeg_data = cv2.imencode('.jpg', frame)
        detections = self.detector.detect(jpeg_data.tobytes())
        
        # 转换检测结果为DeepSORT格式 [x,y,w,h,conf]
        ds_detections = []
        for det in detections:
            bbox = det['bbox']
            ds_detections.append([[bbox[0], bbox[1], bbox[2], bbox[3]], det['conf'], None])
        
        # 更新跟踪器
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)
        
        # 处理跟踪结果
        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = str(track.track_id)
            bbox = track.to_ltrb()  # [left, top, right, bottom]
            
            # 记录轨迹
            center = ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(center)
            
            results.append({
                'track_id': track_id,
                'bbox': [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]],  # [x,y,w,h]
                'confidence': track.get_det_conf()
            })
        
        # 轨迹预测 (线性回归)
        for track_id, points in self.track_history.items():
            if len(points) > 5:  # 有足够历史点才预测
                # 简单线性预测下一帧位置 
                x = np.array([p[0] for p in points[-5:]])
                y = np.array([p[1] for p in points[-5:]])
                
                # 线性回归预测
                x_pred = self.next_frame + 1
                predicted_x = np.poly1d(np.polyfit(range(5), x, 1))(x_pred)
                predicted_y = np.poly1d(np.polyfit(range(5), y, 1))(x_pred)
                
                # 存储预测结果
                points.append((int(predicted_x), int(predicted_y)))
        
        # 绘制可视化结果
        vis_frame = self._visualize(frame, results)
        
        self.next_frame += 1
        
        # 清理过时轨迹
        self._cleanup_history()
        
        return results, vis_frame
    
    def _visualize(self, frame: np.ndarray, tracks: List[Dict]) -> np.ndarray:
        """绘制跟踪可视化"""
        vis = frame.copy()
        
        # 绘制当前检测框
        for track in tracks:
            x,y,w,h = track['bbox']
            cv2.rectangle(vis, (int(x),int(y)), (int(x+w),int(y+h)), (0,255,0), 2)
            cv2.putText(vis, f"ID:{track['track_id']}", (int(x),int(y)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            
            # 绘制轨迹
            if track['track_id'] in self.track_history:
                points = self.track_history[track['track_id']]
                for i in range(1, len(points)):
                    cv2.line(vis, points[i-1], points[i], (0,0,255), 2)
        
        return vis
    
    def _cleanup_history(self):
        """清理过时轨迹"""
        active_ids = {str(track_id) for track_id in self.tracker.tracker.tracks 
                     if self.tracker.tracker.tracks[track_id].is_confirmed()}
        
        # 移除不活跃的轨迹
        for track_id in list(self.track_history.keys()):
            if track_id not in active_ids:
                del self.track_history[track_id]
    
    def __del__(self):
        pass  # 新版DeepSort无需显式清理

# 示例用法
if __name__ == '__main__':
    detector = ConduitDetector()
    tracker = ConduitTracker(detector)
    
    # 模拟视频帧循环
    cap = cv2.VideoCapture('test.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results, vis_frame = tracker.update(frame)
        
        cv2.imshow('Tracking', vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
