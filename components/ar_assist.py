import json
import time
from typing import Dict, List, Optional
from components.data_manage import ARDatabase

class ARAssistProcessor:
    """AR辅助装配处理器"""
    
    def __init__(self, db: ARDatabase, config: Dict):
        self.db = db
        self.config = config
        self.assembly_steps = self.load_assembly_steps()
        
    def load_assembly_steps(self) -> List[Dict]:
        """加载装配步骤指导"""
        # 从数据库加载步骤
        product_id = self.config.get("product_id", "default")
        return self.db.get_assembly_steps(product_id)
    
    def get_current_step(self, session_id: str) -> Optional[Dict]:
        """获取当前装配步骤"""
        # 获取已完成步骤数
        completed_steps = self.db.get_assembly_progress(session_id)
        
        # 获取所有步骤并按顺序排序
        all_steps = sorted(self.assembly_steps, key=lambda x: x["step_order"])
        
        if completed_steps < len(all_steps):
            return all_steps[completed_steps]
        return None
        
    def verify_step_completion(self, session_id: str, tracking_data: Dict) -> bool:
        """验证步骤完成条件"""
        current_step = self.get_current_step(session_id)
        if not current_step:
            return False
            
        # 检查所有条件是否满足
        conditions = current_step["check_conditions"]
        if not isinstance(conditions, list):
            conditions = json.loads(conditions)
            
        for condition in conditions:
            if not self._check_condition(condition, tracking_data):
                return False
                
        # 标记步骤完成
        step_id = current_step["step_id"]
        progress_id = self.db.start_assembly_step(session_id, step_id)
        self.db.complete_assembly_step(progress_id)
        
        return True
        
    def _check_condition(self, condition: str, tracking_data: Dict) -> bool:
        """检查单个条件"""
        # 实际实现中会根据条件类型调用不同的验证方法
        # 这里简化为总是返回True
        return True
        
    def check_part_position(self, tracking_data: Dict, part_id: str) -> bool:
        """检查零件位置"""
        # 实际实现中会使用跟踪数据验证
        return True
        
    def check_alignment(self, tracking_data: Dict, part1: str, part2: str) -> bool:
        """检查零件对齐"""
        # 实际实现中会使用跟踪数据验证
        return True
        
    def generate_ar_instructions(self, session_id: str, tracking_data: Dict) -> Dict:
        """生成AR指令"""
        current_step = self.get_current_step(session_id)
        if not current_step:
            return {"status": "complete"}
            
        return {
            "status": "in_progress",
            "current_step": current_step["step_id"],
            "instruction": current_step["instruction"],
            "visual_cue": current_step["visual_cue"],
            "tracking_data": tracking_data
        }
