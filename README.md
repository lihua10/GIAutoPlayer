# **《原神》自动化脚本开发文档**
### **版本：1.0**  
### **最后更新：2025年1月30日**


## **1. 项目概述**
### **1.1 目标**  
开发一款基于多模态模型Janus-Pro-1B和DeepSeek-1.5B的《原神》全自动脚本，实现以下功能：  
- 通过自然语言解析用户任务指令（如“自动完成每日委托”）。  
- 实时分析游戏画面，生成键鼠操作指令。  
- 模拟人类操作模式执行任务，支持连续性和反检测机制。  

### **1.2 技术栈**  
- **模型框架**：Janus-Pro-1B（视觉推理）、DeepSeek-1.5B（指令解析）  
- **图像处理**：OpenCV、PIL、mss  
- **自动化控制**：pyautogui、pynput  
- **性能优化**：TensorRT、ONNX Runtime、CUDA加速  
- **开发语言**：Python 3.10+  

---

## **2. 系统架构设计**
### **2.1 模块划分**  
1. **指令解析模块**：解析用户自然语言指令，生成结构化API。  
2. **视觉推理模块**：实时分析游戏画面，输出操作指令。  
3. **操作执行模块**：执行键鼠操作，模拟人类行为。  
4. **状态管理模块**：缓存历史数据，保障操作连续性。  
5. **异常处理模块**：超时重置、指令校验、反检测逻辑。  

### **2.2 数据流**  
1. 用户输入自然语言指令 → DeepSeek-1.5B生成JSON格式任务描述。  
2. 游戏画面捕获 → Janus-Pro-1B推理生成操作指令。  
3. 操作指令队列 → 执行模块控制键鼠。  

---

## **3. 核心模块实现细节**
### **3.1 指令解析模块**  
#### **输入输出规范**  
- **输入**：自然语言字符串（例：“自动攻击最近的敌人并收集掉落物”）。  
- **输出**：结构化JSON指令（强制Schema校验）：  
  ```json
  {
    "task_type": "combat",
    "actions": [
      {"type": "attack", "target": "nearest_enemy", "priority": 1},
      {"type": "collect", "target": "loot", "priority": 2}
    ]
  }
  ```

#### **模型微调**  
1. **数据集构建**：  
   - 收集《原神》任务描述与对应API指令的配对数据（例：5000条）。  
   - 标注字段：`task_type`, `actions`, `target`, `priority`。  
2. **训练脚本**：  
   ```python
   from transformers import AutoModelForCausalLM, TrainingArguments

   model = AutoModelForCausalLM.from_pretrained("deepseek-1.5B")
   training_args = TrainingArguments(
       output_dir="output",
       per_device_train_batch_size=4,
       fp16=True,  # RTX 3060支持FP16加速
       gradient_accumulation_steps=2
   )
   # 使用Peft库进行LoRA微调
   model = prepare_model_for_int8_training(model)
   model = get_peft_model(model, lora_config)
   ```

---

### **3.2 视觉推理模块**  
#### **图像预处理**  
1. **屏幕捕获**：  
   ```python
   import mss
   with mss.mss() as sct:
       monitor = sct.monitors[1]  # 主显示器
       img = sct.grab(monitor)
       img = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
   ```
2. **分辨率适配**：动态调整输入分辨率（默认256x256，战斗场景384x384）。  
   ```python
   from PIL import Image
   img = img.resize((256, 256)) if is_low_load else img.resize((384, 384))
   ```

#### **模型推理优化**  
1. **TensorRT部署**：  
   ```bash
   trtexec --onnx=janus_pro.onnx --saveEngine=janus_pro.trt --fp16
   ```
2. **推理脚本**：  
   ```python
   import tensorrt as trt
   runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
   with open("janus_pro.trt", "rb") as f:
       engine = runtime.deserialize_cuda_engine(f.read())
   context = engine.create_execution_context()
   # 绑定输入输出缓冲区
   inputs, outputs, bindings, stream = allocate_buffers(engine)
   ```

---

### **3.3 操作执行模块**  
#### **键鼠控制**  
1. **基础操作**：  
   ```python
   import pyautogui
   def execute_action(action):
       if action["type"] == "move":
           x = action["x"] + random.uniform(-0.5, 0.5)  # 非整数偏移
           y = action["y"] + random.uniform(-0.5, 0.5)
           pyautogui.moveTo(x, y, duration=0.2, tween=pyautogui.easeOutQuad)
       elif action["type"] == "attack":
           pyautogui.click(button="left", _pause=False)
   ```
2. **反检测机制**：  
   - 随机延迟：`time.sleep(random.uniform(0.05, 0.15))`  
   - 贝塞尔曲线移动：  
     ```python
     def bezier_move(start, end, control_points, duration=0.3):
         # 实现自定义轨迹
     ```

---

### **3.4 状态管理模块**  
#### **时间窗口机制**  
1. **数据结构**：  
   ```python
   from collections import deque
   history = deque(maxlen=5)  # 缓存最近5帧数据
   history.append({
       "frame": current_frame,
       "action": last_action,
       "timestamp": time.time()
   })
   ```
2. **上下文拼接**：  
   ```python
   def build_prompt(history):
       context = "|".join([f"[T-{i}] {h['action']}" for i, h in enumerate(history)])
       return f"历史操作：{context} | 当前目标：{current_task}"
   ```

---

## **4. 性能优化方案**
### **4.1 多线程流水线**  
```python
import threading
from queue import Queue

# 全局队列
frame_queue = Queue(maxsize=10)
action_queue = Queue()

# 图像捕获线程
class CaptureThread(threading.Thread):
    def run(self):
        while running:
            frame = capture_screen()
            frame_queue.put(frame)

# 模型推理线程
class InferenceThread(threading.Thread):
    def run(self):
        while running:
            frame = frame_queue.get()
            action = janus_pro.predict(frame)
            action_queue.put(action)

# 启动线程
capture_thread = CaptureThread()
inference_thread = InferenceThread()
capture_thread.start()
inference_thread.start()
```

### **4.2 显存管理**  
- **动态卸载模型**：DeepSeek-1.5B在任务解析完成后立即释放显存。  
  ```python
  import torch
  del deepseek_model
  torch.cuda.empty_cache()
  ```

---

## **5. 部署与测试**
### **5.1 打包部署**  
1. **PyInstaller配置**：  
   ```bash
   pyinstaller --onefile --add-data "models/*;models" app.py
   ```
2. **GUI界面**：使用PyQt5设计任务配置面板：  
   ```python
   from PyQt5.QtWidgets import QLineEdit, QPushButton
   class TaskConfigUI(QWidget):
       def __init__(self):
           self.task_input = QLineEdit("输入任务指令...")
           self.start_btn = QPushButton("开始运行")
   ```

### **5.2 测试计划**  
| 测试类型     | 方法                                                     |
| ------------ | -------------------------------------------------------- |
| **单元测试** | 验证指令解析、图像预处理、动作执行等独立模块的功能正确性 |
| **性能测试** | 使用`py-spy`分析帧率、显存占用和GPU利用率                |
| **合规测试** | 运行24小时检测是否触发游戏反作弊系统                     |

---

## **6. 风险与合规声明**
### **6.1 风险提示**  
- 脚本启动时显示警告：  
  ```text
  [警告] 自动化操作可能导致账号封禁，请谨慎使用！开发者不承担任何责任。
  ```
- 禁止功能清单：PVP战斗、副本速通、资源无限刷取。  

### **6.2 开源协议**  
- 核心框架遵循MIT协议，反检测模块闭源。  

---

## **7. 附录**  
- **参考链接**  
  - Janus-Pro官方仓库：https://github.com/deepseek-ai/Janus  
  - DeepSeek-1.5B模型：https://huggingface.co/deepseek-ai/deepseek-1.5B  
- **支持联系**：service@deepseek.com  

---

**文档修订历史**  
| 版本 | 修改内容 | 日期       |
| ---- | -------- | ---------- |
| 1.0  | 初稿发布 | 2023-10-01 |

---

通过本文档，开发团队可系统性完成从模型集成到自动化控制的完整流程，确保项目高效、稳定地运行。