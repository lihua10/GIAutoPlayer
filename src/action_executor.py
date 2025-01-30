import time
import random
import threading
import queue
from typing import List, Dict, Any
from pynput import mouse, keyboard
import numpy as np
from scipy.interpolate import make_interp_spline

class ActionExecutor:
    def __init__(self):
        self.action_queue = queue.Queue()
        self.running = False
        self.mouse_controller = mouse.Controller()
        self.keyboard_controller = keyboard.Controller()
        self.current_window_focused = True
        
    def start(self):
        """启动动作执行线程"""
        self.running = True
        self.executor_thread = threading.Thread(target=self._process_actions)
        self.executor_thread.daemon = True
        self.executor_thread.start()
        
    def stop(self):
        """停止动作执行"""
        self.running = False
        self.executor_thread.join()
        
    def add_action(self, action: Dict[str, Any]):
        """添加动作到队列"""
        self.action_queue.put(action)
        
    def _process_actions(self):
        """处理动作队列"""
        while self.running:
            if not self.current_window_focused:
                time.sleep(0.1)
                continue
                
            if not self.action_queue.empty():
                action = self.action_queue.get()
                self._execute_action(action)
                
            time.sleep(0.01)
            
    def _execute_action(self, action: Dict[str, Any]):
        """执行单个动作"""
        action_type = action.get("type")
        
        if action_type == "mouse_move":
            self._mouse_move(action["target"])
        elif action_type == "mouse_click":
            self._mouse_click(action["button"])
        elif action_type == "key_press":
            self._key_press(action["key"])
        elif action_type == "key_combination":
            self._key_combination(action["keys"])
            
    def _mouse_move(self, target: List[int]):
        """鼠标移动"""
        current_pos = self.mouse_controller.position
        target_pos = tuple(target)
        
        # 生成贝塞尔曲线路径
        points = self._generate_bezier_path(current_pos, target_pos)
        
        # 平滑移动
        for point in points:
            self.mouse_controller.position = point
            time.sleep(random.uniform(0.001, 0.01))
            
    def _mouse_click(self, button: str):
        """鼠标点击"""
        self.mouse_controller.click(mouse.Button[button])
        time.sleep(random.uniform(0.05, 0.15))  # 随机延迟
        
    def _key_press(self, key: str):
        """按键"""
        self.keyboard_controller.press(key)
        time.sleep(random.uniform(0.05, 0.15))
        self.keyboard_controller.release(key)
        
    def _key_combination(self, keys: List[str]):
        """组合键"""
        for key in keys:
            self.keyboard_controller.press(key)
            time.sleep(random.uniform(0.05, 0.15))
            
        for key in reversed(keys):
            self.keyboard_controller.release(key)
            time.sleep(random.uniform(0.05, 0.15))
            
    def _generate_bezier_path(self, start: tuple, end: tuple, num_points: int = 20) -> List[tuple]:
        """生成贝塞尔曲线路径"""
        # 生成控制点
        control_point1 = (
            start[0] + (end[0] - start[0]) * 0.3,
            start[1] + (end[1] - start[1]) * 0.7
        )
        control_point2 = (
            start[0] + (end[0] - start[0]) * 0.7,
            start[1] + (end[1] - start[1]) * 0.3
        )
        
        # 生成曲线
        t = np.linspace(0, 1, num_points)
        x = (1-t)**3 * start[0] + 3*(1-t)**2*t*control_point1[0] + 3*(1-t)*t**2*control_point2[0] + t**3*end[0]
        y = (1-t)**3 * start[1] + 3*(1-t)**2*t*control_point1[1] + 3*(1-t)*t**2*control_point2[1] + t**3*end[1]
        
        return list(zip(x.astype(int), y.astype(int)))
        
    def on_window_focus(self, focused: bool):
        """窗口焦点变化回调"""
        self.current_window_focused = focused

if __name__ == "__main__":
    # 测试代码
    executor = ActionExecutor()
    executor.start()
    
    # 添加测试动作
    executor.add_action({
        "type": "mouse_move",
        "target": [500, 500]
    })
    executor.add_action({
        "type": "mouse_click",
        "button": "left"
    })
    executor.add_action({
        "type": "key_press",
        "key": "a"
    })
    
    time.sleep(5)
    executor.stop()
