import cv2
import numpy as np

# 初始化ORB特征点检测器
orb = cv2.ORB_create()

# 创建BFMatcher对象，用于特征点匹配
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 加载参考图像（静态物体的图像）
img1 = cv2.imread('object_image.jpg', 0)  # 读取为灰度图像
if img1 is None:
    print("图像加载失败")
    exit()

# 计算参考图像的ORB特征点和描述符
kp1, des1 = orb.detectAndCompute(img1, None)

# 创建视频捕捉对象
cap = cv2.VideoCapture(0)  # 使用摄像头进行实时捕捉
if not cap.isOpened():
    print("摄像头无法打开")
    exit()

while True:
    ret, frame = cap.read()  # 从摄像头读取一帧图像
    if not ret:
        print("无法读取视频帧")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像

    # 检测当前帧的特征点和描述符
    kp2, des2 = orb.detectAndCompute(gray, None)

    # 如果当前帧的描述符为空，跳过该帧
    if des2 is None:
        continue

    # 匹配当前帧和参考图像的特征点
    matches = bf.match(des1, des2)

    # 按照匹配距离排序
    matches = sorted(matches, key = lambda x: x.distance)

    # 绘制前20个匹配点
    img_matches = cv2.drawMatches(img1, kp1, frame, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 显示匹配结果
    cv2.imshow('Feature Matches', img_matches)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源并关闭所有窗口
cap.release()
cv2.destroyAllWindows()