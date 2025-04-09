import sqlite3
import uuid
import json
from typing import Optional, Dict, List
from dataclasses import dataclass

@dataclass
class ARObject:
    """目标对象实体类"""
    object_id: str  # UUID
    type: str       # 导管型号
    model_path: str # 3D模型路径
    dimensions: Dict[str, float]  # 长宽高

@dataclass 
class TrackingRecord:
    """跟踪记录实体类"""
    track_id: int   # 自增ID
    object_id: str  # 外键
    timestamp: str  # ISO格式时间
    camera_coords: Dict[str, float]  # {x,y,w,h}
    world_coords: Dict[str, float]   # {x,y,z}

class ARDatabase:
    """AR装配跟踪数据库管理类"""
    
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        
    def _init_db(self):
        """初始化数据库表结构"""
        with self.conn:
            # 启用外键约束
            self.conn.execute("PRAGMA foreign_keys = ON")
            
            # 创建目标对象表
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS objects (
                object_id TEXT PRIMARY KEY,
                type VARCHAR(50) NOT NULL,
                model_path TEXT NOT NULL,
                dimensions TEXT NOT NULL
            )""")
            
            # 创建跟踪记录表
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tracking (
                track_id INTEGER PRIMARY KEY AUTOINCREMENT,
                object_id TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                camera_coords TEXT NOT NULL,
                world_coords TEXT NOT NULL,
                FOREIGN KEY (object_id) REFERENCES objects(object_id)
            )""")
            
            # 创建图像元数据表
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                image_hash CHAR(32) PRIMARY KEY,
                frame_time DATETIME NOT NULL,
                raw_data BLOB NOT NULL
            )""")
            
            # 创建空间索引(R*Tree)加速坐标查询
            self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS coord_index USING rtree(
                id,              -- 主键
                minX, maxX,      -- X轴范围
                minY, maxY,      -- Y轴范围
                minZ, maxZ       -- Z轴范围
            )""")
            
            # 创建装配会话表
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS assembly_sessions (
                session_id TEXT PRIMARY KEY,
                product_id TEXT NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                status TEXT CHECK(status IN ('pending', 'in_progress', 'completed', 'paused'))
            )""")
            
            # 创建装配步骤表
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS assembly_steps (
                step_id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT NOT NULL,
                step_order INTEGER NOT NULL,
                name TEXT NOT NULL,
                instruction TEXT NOT NULL,
                visual_cue TEXT NOT NULL,
                check_conditions TEXT NOT NULL  -- JSON数组
            )""")
            
            # 创建装配进度表
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS assembly_progress (
                progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                step_id INTEGER NOT NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME,
                status TEXT CHECK(status IN ('pending', 'in_progress', 'completed', 'failed')),
                FOREIGN KEY (session_id) REFERENCES assembly_sessions(session_id),
                FOREIGN KEY (step_id) REFERENCES assembly_steps(step_id)
            )""")
    
    def add_object(self, obj: ARObject) -> str:
        """添加目标对象"""
        with self.conn:
            obj_id = str(uuid.uuid4())
            self.conn.execute(
                "INSERT INTO objects VALUES (?, ?, ?, ?)",
                (obj_id, obj.type, obj.model_path, json.dumps(obj.dimensions))
            )
            return obj_id
    
    def add_tracking(self, record: TrackingRecord) -> int:
        """添加跟踪记录"""
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO tracking VALUES (NULL, ?, ?, ?, ?) RETURNING track_id",
                (
                    record.object_id,
                    record.timestamp,
                    json.dumps(record.camera_coords),
                    json.dumps(record.world_coords)
                )
            )
            track_id = cursor.fetchone()[0]
            
            # 更新空间索引
            coords = record.world_coords
            self.conn.execute(
                "INSERT INTO coord_index VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    track_id,
                    coords['x'], coords['x'],
                    coords['y'], coords['y'],
                    coords['z'], coords['z']
                )
            )
            return track_id
    
    def get_objects_in_radius(self, x: float, y: float, z: float, radius: float) -> List[TrackingRecord]:
        """查询指定半径内的跟踪记录"""
        with self.conn:
            cursor = self.conn.execute("""
                SELECT t.* FROM tracking t
                JOIN coord_index c ON c.id = t.track_id
                WHERE c.minX BETWEEN ? AND ?
                AND c.minY BETWEEN ? AND ?
                AND c.minZ BETWEEN ? AND ?
            """, (
                x - radius, x + radius,
                y - radius, y + radius,
                z - radius, z + radius
            ))
            
            return [
                TrackingRecord(
                    track_id=row[0],
                    object_id=row[1],
                    timestamp=row[2],
                    camera_coords=json.loads(row[3]),
                    world_coords=json.loads(row[4])
                )
                for row in cursor.fetchall()
            ]

    # AR装配相关方法
    def create_assembly_session(self, product_id: str) -> str:
        """创建新的装配会话"""
        session_id = str(uuid.uuid4())
        with self.conn:
            self.conn.execute(
                "INSERT INTO assembly_sessions VALUES (?, ?, datetime('now'), NULL, 'pending')",
                (session_id, product_id)
            )
        return session_id

    def get_assembly_steps(self, product_id: str) -> List[Dict]:
        """获取产品的装配步骤"""
        with self.conn:
            cursor = self.conn.execute(
                "SELECT * FROM assembly_steps WHERE product_id = ? ORDER BY step_order",
                (product_id,)
            )
            return [
                {
                    "step_id": row[0],
                    "product_id": row[1],
                    "step_order": row[2],
                    "name": row[3],
                    "instruction": row[4],
                    "visual_cue": row[5],
                    "check_conditions": json.loads(row[6])
                }
                for row in cursor.fetchall()
            ]

    def start_assembly_step(self, session_id: str, step_id: int) -> int:
        """开始装配步骤"""
        with self.conn:
            cursor = self.conn.execute(
                """INSERT INTO assembly_progress VALUES 
                (NULL, ?, ?, datetime('now'), NULL, 'in_progress') 
                RETURNING progress_id""",
                (session_id, step_id)
            )
            return cursor.fetchone()[0]

    def complete_assembly_step(self, progress_id: int):
        """完成装配步骤"""
        with self.conn:
            self.conn.execute(
                "UPDATE assembly_progress SET end_time = datetime('now'), status = 'completed' WHERE progress_id = ?",
                (progress_id,)
            )

    def get_assembly_progress(self, session_id: str) -> int:
        """获取当前装配进度(已完成步骤数)"""
        with self.conn:
            cursor = self.conn.execute(
                "SELECT COUNT(*) FROM assembly_progress WHERE session_id = ? AND status = 'completed'",
                (session_id,)
            )
            return cursor.fetchone()[0]

# 示例DDL导出
DDL_STATEMENTS = [
    """
    CREATE TABLE objects (
        object_id TEXT PRIMARY KEY,
        type VARCHAR(50) NOT NULL,
        model_path TEXT NOT NULL,
        dimensions TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE tracking (
        track_id INTEGER PRIMARY KEY AUTOINCREMENT,
        object_id TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        camera_coords TEXT NOT NULL,
        world_coords TEXT NOT NULL,
        FOREIGN KEY (object_id) REFERENCES objects(object_id)
    )
    """,
    """
    CREATE VIRTUAL TABLE coord_index USING rtree(
        id, minX, maxX, minY, maxY, minZ, maxZ
    )
    """,
    """
    CREATE TABLE assembly_sessions (
        session_id TEXT PRIMARY KEY,
        product_id TEXT NOT NULL,
        start_time DATETIME NOT NULL,
        end_time DATETIME,
        status TEXT CHECK(status IN ('pending', 'in_progress', 'completed', 'paused'))
    )
    """,
    """
    CREATE TABLE assembly_steps (
        step_id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id TEXT NOT NULL,
        step_order INTEGER NOT NULL,
        name TEXT NOT NULL,
        instruction TEXT NOT NULL,
        visual_cue TEXT NOT NULL,
        check_conditions TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE assembly_progress (
        progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        step_id INTEGER NOT NULL,
        start_time DATETIME NOT NULL,
        end_time DATETIME,
        status TEXT CHECK(status IN ('pending', 'in_progress', 'completed', 'failed')),
        FOREIGN KEY (session_id) REFERENCES assembly_sessions(session_id),
        FOREIGN KEY (step_id) REFERENCES assembly_steps(step_id)
    )
    """
]
