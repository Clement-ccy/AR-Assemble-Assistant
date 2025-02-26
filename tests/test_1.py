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

# 对应的图像 2D 点
img_points = np.array([
    [100, 100],
    [200, 100],
    [100, 200],
    [200, 200],
    [100, 300],
    [200, 300]
], dtype=np.float32)

# 姿态估计
success, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)

if success:
    print("Rotation Vector (rvec):\n", rvec)
    print("Translation Vector (tvec):\n", tvec)

    # 投影 3D 点到图像平面
    projected_points, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)

    # 创建示例图像
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 绘制点
    for p in projected_points:
        center = tuple(map(int, p.ravel()))  # 转换为整数
        cv2.circle(frame, center, 5, (0, 255, 0), -1)

    cv2.imshow('3D Model Tracking', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Pose estimation failed.")