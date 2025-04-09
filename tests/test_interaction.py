import socket
import json
import random
import time
import math

host = '127.0.0.1'
port = 12345

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((host, port))
    s.listen()
    print("Waiting for Unity connection...")
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        while True:
            # 生成模拟检测数据（归一化坐标）
            data = {
                # 生成更自然的运动模拟（余弦波动）
                "x": (math.cos(time.time() * 2) + 1) / 2,  # 生成0-1的波动
                "y": (math.sin(time.time() * 2) + 1) / 2    # 生成0-1的波动
            }
            
            # 发送JSON数据并添加换行符
            json_data = json.dumps(data) + '\n'
            print(data)
            conn.sendall(json_data.encode('utf-8'))
            time.sleep(0.1)  # 控制发送频率