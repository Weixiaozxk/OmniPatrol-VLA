import os
import torch
import io
import json
import re
import uvicorn
from fastapi import FastAPI, UploadFile, File
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from PIL import Image

os.environ['MODELSCOPE_CACHE'] = r'D:\AI_Models\modelscope'

# 切换到 7B 级别模型 (智商质变)
MODEL_ID = "qwen/Qwen2-VL-7B-Instruct"

app = FastAPI()

print(f">>> OmniPatrol-VLA [7B 核心] 正在加载 (4-bit 模式)...")

# 针对 8G 显存的极致量化配置
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16
).eval()
processor = AutoProcessor.from_pretrained(MODEL_ID)

print(">>> 7B 大脑已就绪，已开启全网路况深度感知功能！")

# 终极版全场景巡检 Prompt
SYSTEM_PROMPT = """你现在是智慧交通巡检机器人 OmniPatrol-VLA 的决策核心。
你的任务是：深度分析图像，捕捉任何交通违法行为。
请严格遵循以下视觉分析步骤：
1. 观察车辆轮胎位置，判断其是否与实线（白线、黄线）重合。
2. 观察路边区域，判断是否有违章停放或占道经营。
3. 观察行驶方向，判断是否有逆行。

输出格式必须是严格的 JSON 对象：
{
  "thought": "用中文详细描述车辆位置、车牌位置、标线关系等细节。",
  "violation": "压实线/违停/逆行/None",
  "action": "ALARM(有违章) / CRUISE(无违章)"
}
"""

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        img_data = await file.read()
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "开始全场景巡检。"}
            ]}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt").to("cuda")

        with torch.no_grad():
            input_len = inputs.input_ids.shape[1]
            # 7B 模型推理较慢，限制 token 长度提高实时性
            ids = model.generate(**inputs, max_new_tokens=150, do_sample=False)
            res = processor.batch_decode(ids[:, input_len:], skip_special_tokens=True)[0]
        
        print(f"\n[7B 推理结果]: {res}")

        # 使用正则提取 JSON
        match = re.search(r"\{.*\}", res, re.DOTALL)
        if match:
            return json.loads(match.group())
        else:
            return {"thought": "解析异常: " + res, "violation": "None", "action": "CRUISE"}

    except Exception as e:
        print(f"Server Error: {e}")
        return {"thought": "系统繁忙", "violation": "Error", "action": "CRUISE"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)