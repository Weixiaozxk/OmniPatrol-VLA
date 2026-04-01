import os
import json
import cv2
import numpy as np

# 1. 定义工作区路径
BASE_DIR = r"E:\OmniPatrol-VLA\finetune"
DATA_DIR = os.path.join(BASE_DIR, "data")
IMG_DIR = os.path.join(DATA_DIR, "images")

os.makedirs(IMG_DIR, exist_ok=True)

print(">>> 正在为你构建 OmniPatrol-VLA 微调工作区...")

# 2. 生成模拟测试图片 (防止你没有图导致报错，生成几张带白线/黄线的图)
# 真实训练时，你可以把真实的违章图扔进这个 images 文件夹替换它们
for i in range(1, 4):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    if i == 1:
        cv2.line(img, (100, 100), (500, 400), (255, 255, 255), 5) # 白实线
        cv2.putText(img, "Car on White Line", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    elif i == 2:
        cv2.line(img, (100, 400), (500, 100), (0, 255, 255), 5) # 黄实线
        cv2.putText(img, "Car on Yellow Line", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    else:
        cv2.putText(img, "Normal Road, No lines touched", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    cv2.imwrite(os.path.join(IMG_DIR, f"sample_{i}.jpg"), img)

# 3. 自动生成 LLaMA-Factory 数据集 (ShareGPT 格式)
dataset =[
    {
        "messages":[
            {"role": "user", "content": "<image>作为智慧交通交警，分析此画面。"},
            {"role": "assistant", "content": "{\"thought\": \"画面中车辆压过白色实线，违反禁止标线指示。\", \"violation\": \"压实线违章\", \"action\": \"ALARM\"}"}
        ],
        "images": ["images/sample_1.jpg"]
    },
    {
        "messages":[
            {"role": "user", "content": "<image>作为智慧交通交警，分析此画面。"},
            {"role": "assistant", "content": "{\"thought\": \"画面中车辆压过黄色实线，存在逆行或违规超车风险。\", \"violation\": \"压黄实线违章\", \"action\": \"ALARM\"}"}
        ],
        "images": ["images/sample_2.jpg"]
    },
    {
        "messages":[
            {"role": "user", "content": "<image>作为智慧交通交警，分析此画面。"},
            {"role": "assistant", "content": "{\"thought\": \"画面中车辆在车道内正常行驶，未触碰任何实线。\", \"violation\": \"None\", \"action\": \"CRUISE\"}"}
        ],
        "images": ["images/sample_3.jpg"]
    }
]

# 写入数据集文件
with open(os.path.join(DATA_DIR, "traffic_vla.json"), "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

# 4. 生成 dataset_info.json (这是框架识别数据集的钥匙)
dataset_info = {
    "traffic_vla": {
        "file_name": "traffic_vla.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "images"
        }
    }
}
with open(os.path.join(DATA_DIR, "dataset_info.json"), "w", encoding="utf-8") as f:
    json.dump(dataset_info, f, indent=2)

# 5. 生成 4060 专属训练配置文件 (yaml)
yaml_content = """
### model
model_name_or_path: D:/AI_Models/modelscope/models/qwen/Qwen2-VL-2B-Instruct

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all

### dataset
dataset: traffic_vla
dataset_dir: data
template: qwen2_vl
cutoff_len: 512
overwrite_cache: true
preprocessing_num_workers: 1

### output
output_dir: ../omnipatrol_lora_model
logging_steps: 1
save_steps: 50
plot_loss: true

### train (4060 8GB 极致优化)
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
num_train_epochs: 15.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
fp16: true
quantization_bit: 4
"""
with open(os.path.join(BASE_DIR, "train_qwen.yaml"), "w", encoding="utf-8") as f:
    f.write(yaml_content.strip())

print("\n>>> 工作区构建成功！")
print("请在终端执行以下命令开始炼丹：")
print(f"cd {BASE_DIR}")
print("llamafactory-cli train train_qwen.yaml")