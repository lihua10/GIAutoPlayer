import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any
import json

class DecisionEngine:
    def __init__(self, model_path: str = "models/DeepSeek_R1_Distill_Qwen_1.5B.safetensors"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("DeepSeek/DeepSeek-R1-Distill-Qwen-1.5B")
        self.temperature = 0.7
        self.max_tokens = 512
        
    def _load_model(self, model_path: str) -> Any:
        """加载DeepSeek模型"""
        try:
            state_dict = load_file(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                "DeepSeek/DeepSeek-R1-Distill-Qwen-1.5B",
                state_dict=state_dict,
                device_map="auto"
            )
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
            
    def load_lora_adapter(self, lora_path: str):
        """加载LoRA适配器"""
        # 这里需要实现LoRA适配器加载逻辑
        pass
        
    def generate_instruction(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        根据游戏状态生成操作指令
        :param game_state: 游戏状态字典
        :return: 操作指令字典
        """
        prompt = self._create_prompt(game_state)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=self.max_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_response(response)
        
    def _create_prompt(self, game_state: Dict[str, Any]) -> str:
        """创建LLM提示词"""
        return f"""根据以下游戏状态生成操作指令：
{json.dumps(game_state, indent=2)}

请以JSON格式返回操作指令，包含以下字段：
- actions: 操作列表
- priority: 操作优先级
- timing: 执行时机
"""
        
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """解析LLM响应为JSON"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果解析失败，返回默认操作
            return {
                "actions": [],
                "priority": 0,
                "timing": "immediate"
            }

if __name__ == "__main__":
    # 测试代码
    engine = DecisionEngine()
    game_state = {
        "血量": 30,
        "敌人位置": [120, 80],
        "技能冷却": [True, False, True]
    }
    instruction = engine.generate_instruction(game_state)
    print(f"Generated instruction: {instruction}")
