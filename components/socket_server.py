import asyncio
import struct
import zlib
import json
import websockets
from typing import Optional, Set
from components.data_manage import ARDatabase

class ARSocketServer:
    """AR装配跟踪Socket服务端(支持TCP和WebSocket)"""
    
    def __init__(self, db: ARDatabase, config: Dict):
        self.db = db
        self.config = config.get('socket', {})
        self.tcp_clients = set()
        self.websocket_clients: Set[websockets.WebSocketServerProtocol] = set()
        
    async def handle_tcp_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """处理TCP客户端连接"""
        addr = writer.get_extra_info('peername')
        print(f"New TCP connection from {addr}")
        self.tcp_clients.add(writer)
        
        try:
            while True:
                # 读取4字节长度头
                header = await reader.readexactly(4)
                if not header:
                    break
                    
                length = struct.unpack('!I', header)[0]
                
                # 读取数据部分
                data = await reader.readexactly(length)
                
                # CRC校验
                crc = struct.unpack('!I', data[-4:])[0]
                payload = data[:-4]
                if zlib.crc32(payload) != crc:
                    print(f"CRC mismatch from {addr}")
                    continue
                
                # 处理图像帧
                track_result = await self.process_frame(payload)
                
                # 返回跟踪结果
                response = json.dumps(track_result).encode()
                writer.write(struct.pack('!I', len(response)) + response)
                await writer.drain()
                
        except (asyncio.IncompleteReadError, ConnectionResetError):
            print(f"Client {addr} disconnected")
        finally:
            self.clients.remove(writer)
            writer.close()
            await writer.wait_closed()
    
    async def process_frame(self, jpeg_data: bytes) -> dict:
        """处理图像帧并返回跟踪结果"""
        # 这里应该是实际的AR跟踪算法
        # 模拟返回结果
        return {
            "track_id": 1,
            "bbox": [100, 100, 200, 200],
            "world_pos": [0.5, 0.2, 1.8]
        }
    
    async def handle_websocket(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """处理WebSocket连接"""
        print(f"New WebSocket connection from {websocket.remote_address}")
        self.websocket_clients.add(websocket)
        try:
            async for message in websocket:
                # 处理来自Hololens的消息
                data = json.loads(message)
                if data.get('type') == 'session_start':
                    session_id = self.db.create_assembly_session(data['product_id'])
                    await websocket.send(json.dumps({
                        'type': 'session_created',
                        'session_id': session_id
                    }))
        except websockets.exceptions.ConnectionClosed:
            print("WebSocket connection closed")
        finally:
            self.websocket_clients.remove(websocket)

    async def broadcast_to_hololens(self, message: Dict):
        """广播消息给所有Hololens客户端"""
        if self.websocket_clients:
            message_str = json.dumps(message)
            await asyncio.gather(
                *[client.send(message_str) for client in self.websocket_clients]
            )

    async def start(self):
        """启动TCP和WebSocket服务器"""
        # 启动TCP服务器
        tcp_server = await asyncio.start_server(
            self.handle_tcp_client,
            self.config.get('host', '0.0.0.0'),
            self.config.get('port', 8888)
        )
        print(f"TCP server started on {self.config.get('host', '0.0.0.0')}:{self.config.get('port', 8888)}")
        
        # 启动WebSocket服务器
        websocket_server = await websockets.serve(
            self.handle_websocket,
            self.config.get('host', '0.0.0.0'),
            self.config.get('hololens_port', 8889)
        )
        print(f"WebSocket server started on {self.config.get('host', '0.0.0.0')}:{self.config.get('hololens_port', 8889)}")
        
        # 同时运行两个服务器
        await asyncio.gather(
            tcp_server.serve_forever(),
            websocket_server.wait_closed()
        )

async def stress_test(concurrent: int = 10):
    """压力测试函数"""
    db = ARDatabase()
    server = ARSocketServer(db)
    
    async def mock_client():
        reader, writer = await asyncio.open_connection('127.0.0.1', 8888)
        try:
            # 模拟发送1280x720的JPEG帧 (50KB)
            mock_data = b'\xff\xd8' + b'\x00' * 50000 + b'\xff\xd9'
            crc = zlib.crc32(mock_data)
            packet = struct.pack('!I', len(mock_data) + 4) + mock_data + struct.pack('!I', crc)
            
            for _ in range(100):  # 每个客户端发送100帧
                writer.write(packet)
                await writer.drain()
                
                # 读取响应
                header = await reader.readexactly(4)
                length = struct.unpack('!I', header)[0]
                response = await reader.readexactly(length)
                json.loads(response.decode())  # 验证JSON格式
                
        finally:
            writer.close()
            await writer.wait_closed()
    
    # 启动服务器任务
    server_task = asyncio.create_task(server.start())
    await asyncio.sleep(1)  # 等待服务器启动
    
    # 启动并发客户端
    tasks = [mock_client() for _ in range(concurrent)]
    await asyncio.gather(*tasks)
    
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        pass

if __name__ == '__main__':
    # 示例用法
    db = ARDatabase()
    server = ARSocketServer(db)
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("Server shutdown")
