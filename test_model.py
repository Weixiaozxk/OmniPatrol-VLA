import os
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
import json # 用于解析模型输出的 JSON 格式

# 1. 环境设定
os.environ['MODELSCOPE_CACHE'] = r'D:\AI_Models\modelscope'
MODEL_DIR = r"D:\AI_Models\modelscope\models\qwen\Qwen2-VL-2B-Instruct"

class OmniPatrolVLA:
    def __init__(self):
        print(">>> 正在激活 OmniPatrol-VLA 大脑核心 (Qwen2-VL-2B-Instruct) ...")
        
        # 4-bit 量化加载 (适用于 RTX 4060 8GB 显存)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self0.model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_DIR,
            quantization_config=quant_config,
            device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(MODEL_DIR)
        
        # --- 核心：通过系统 Prompt 注入“交通警察”逻辑和输出格式 ---
        self.system_prompt = (
            "你叫 OmniPatrol-VLA，是由独立开发者开发的智慧交通多模态巡检机器人。"
            "你的任务是作为一名极其严格且专业的交通巡警，分析监控画面并给出决策。\n"
            "你的判断标准和动作优先级如下：\n"
            "1. **压实线违章 (Solid Line Violation)**：只要车辆的任何轮胎（包括虚压）触碰到白实线或黄实线，即判断为违章。此为高优先级违章。\n"
            "2. **违规停车 (Illegal Parking)**：在非停车区域（如人行道、禁停区、路口附近）长时间停留的车辆，即判断为违章。\n"
            "3. **方向违章 (Direction Violation)**：在规定直行车道进行转弯，或在禁止转弯区域转弯。\n"
            "4. **行人违章 (Pedestrian Violation)**：行人闯红灯、翻越护栏等行为。\n"
            "请仔细观察画面中所有车辆和行人的行为，并重点关注交通标线。\n"
            "你必须严格按照以下 JSON 格式输出响应。如果没有违章，'violation' 字段请填写 'None'，'action' 字段请填写 'CRUISE'。\n"
            '{"thought": "详细的场景分析和违章判断理由", "violation_type": "违章类型/None", "action_command": "CRUISE/STOP_AND_ALARM/CAPTURE_EVIDENCE/SLOW_DOWN"}'
        )
        
        # 动作指令映射表 (用于解析模型输出，并指导实际执行)
        self.action_mapping = {
            "CRUISE": "继续沿预设路径巡航。",
            "STOP_AND_ALARM": "立即停止，并启动声光报警，语音播报违章信息。",
            "CAPTURE_EVIDENCE": "保持当前位置，启动高精度摄像头对违章目标进行拍照取证。",
            "SLOW_DOWN": "降低机器人行驶速度，保持谨慎巡航。"
        }
        print(">>> OmniPatrol-VLA 大脑已就绪，等待指令。")

    def patrol_inference(self, image_path):
        if not os.path.exists(image_path):
            print(f"错误：找不到本地图片文件！路径: {image_path}")
            return None

        image = Image.open(image_path)
        
        # 将系统提示和用户指令合并
        user_input_with_prompt = self.system_prompt + "\n\n请分析当前路面情况，并给出你的决策。"
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_input_with_prompt}
            ],
        }]

        # 处理输入
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

        # 生成结果
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True) # 增加一些采样多样性
            raw_output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 尝试解析 JSON 格式
        try:
            # 找到 JSON 字符串的起始和结束
            json_start = raw_output.find('{')
            json_end = raw_output.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_str = raw_output[json_start:json_end]
                parsed_output = json.loads(json_str)
                # 根据 action_command 查找实际执行指令
                action_desc = self.action_mapping.get(parsed_output.get("action_command", "CRUISE"), "未知指令，默认巡航。")
                parsed_output["action_description"] = action_desc
                return parsed_output
            else:
                return {"error": "模型未输出有效 JSON 格式", "raw_output": raw_output}
        except json.JSONDecodeError:
            return {"error": "模型输出的 JSON 格式不正确", "raw_output": raw_output}

# --- 运行测试 ---
if __name__ == "__main__":
    vla = OmniPatrolVLA()
    
    # 放置你的本地交通图片，或者你可以临时用一张网络图测试
    # 比如从 ModelScope 找一张违章图片：
    # test_img_path = "D:\\AI_Models\\modelscope\\models\\qwen\\Qwen2-VL-2B-Instruct\\traffic_violation_example.jpg"
    # 或者你自己保存一张到 E:\OmniPatrol-VLA\traffic_test.jpg
    test_img_path = "E:\\OmniPatrol-VLA\\traffic_test.jpg" 
    
    print("\n" + "="*20 + " 正在执行智能巡检推理 " + "="*20)
    output_data = vla.patrol_inference(test_img_path)
    
    if output_data:
        print(json.dumps(output_data, indent=4, ensure_ascii=False))
        # 示例：提取并打印动作指令
        if "action_command" in output_data:
            print(f"\n机器人应执行的动作: {output_data['action_command']} ({output_data.get('action_description', '未定义')})")
    else:
        print("未能获取有效推理结果。")
    print("="*60)