import yaml
import json
from components.data_manage import ARDatabase

def init_database(config_path='config.yaml'):
    # 加载配置
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 初始化数据库
    db = ARDatabase(config['database']['path'])
    
    # 添加初始装配步骤
    for step in config['ar_assist']['initial_steps']:
        db.conn.execute(
            """INSERT INTO assembly_steps 
            (product_id, step_order, name, instruction, visual_cue, check_conditions)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                config['ar_assist']['product_id'],
                step['step_order'],
                step['name'],
                step['instruction'],
                step['visual_cue'],
                json.dumps(step['check_conditions'])
            )
        )
    
    print("Database initialized successfully")

if __name__ == '__main__':
    init_database()
