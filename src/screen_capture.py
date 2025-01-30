import time
import mss
import mss.tools
from PIL import Image
from typing import Tuple, Optional

class ScreenCapture:
    def __init__(self, monitor: int = 1, fps: int = 30):
        self.monitor = monitor
        self.fps = fps
        self.sct = mss.mss()
        self.last_capture_time = 0
        self.frame_interval = 1.0 / fps
        
    def get_monitor_resolution(self) -> Tuple[int, int]:
        """获取指定显示器的分辨率"""
        monitor_info = self.sct.monitors[self.monitor]
        return monitor_info['width'], monitor_info['height']
        
    def capture(self, retry_count: int = 3) -> Optional[Image.Image]:
        """捕获屏幕图像，支持重试机制"""
        for attempt in range(retry_count):
            try:
                current_time = time.time()
                if current_time - self.last_capture_time < self.frame_interval:
                    time.sleep(self.frame_interval - (current_time - self.last_capture_time))
                    
                monitor = self.sct.monitors[self.monitor]
                sct_img = self.sct.grab(monitor)
                img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)
                self.last_capture_time = time.time()
                return img
            except Exception as e:
                print(f"Capture failed (attempt {attempt + 1}/{retry_count}): {str(e)}")
                time.sleep(0.1)
        return None

    def adaptive_capture(self, target_size: Tuple[int, int]) -> Optional[Image.Image]:
        """自适应分辨率捕获"""
        img = self.capture()
        if img is None:
            return None
            
        # 如果分辨率不匹配则进行缩放
        if img.size != target_size:
            img = img.resize(target_size, Image.Resampling.LANCZOS)
        return img

if __name__ == "__main__":
    # 测试代码
    capture = ScreenCapture()
    img = capture.capture()
    if img:
        print(f"Capture successful, image size: {img.size}")
        img.save("test_capture.png")
    else:
        print("Capture failed")
