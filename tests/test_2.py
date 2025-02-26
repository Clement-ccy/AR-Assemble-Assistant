import cv2
import numpy as np

# 摄像机内参和畸变系数
camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))  # 假设无畸变

# 3D 模型关键点（示例数据）
obj_points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1]
], dtype=np.float32)

# 对应的图像 2D 点（根据实际情况动态获取或假设一些点）
img_points = np.array([
    [100, 100],
    [200, 100],
    [100, 200],
    [200, 200],
    [100, 300],
    [200, 300]
], dtype=np.float32)

# 启动摄像头
cap = cv2.VideoCapture(0)  # 默认摄像头
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # 假设我们有对应的 2D 点 img_points，这里可以根据需求动态更新
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)

    if success:
        # 投影 3D 点到图像平面
        projected_points, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)

        # 绘制点
        for p in projected_points:
            center = tuple(map(int, p.ravel()))  # 转换为整数
            cv2.circle(frame, center, 5, (0, 255, 0), -1)

        # 显示坐标系
        cv2.putText(frame, "3D Tracking Active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        # 如果姿态估计失败，提示用户
        cv2.putText(frame, "Pose estimation failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 显示实时视频流
    cv2.imshow('3D Model Tracking', frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()