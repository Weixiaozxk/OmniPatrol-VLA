import cv2
import os
import json
import random
import numpy as np

# --- 路径配置 ---
RAW_DIR = r"E:\OmniPatrol-VLA\finetune\data\raw_images" # 你放10张图的地方
BASE_DIR = r"E:\OmniPatrol-VLA\finetune\data"
IMG_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

print(">>> 🚀 启动 OmniPatrol-VLA 数据工厂...")

def augment_image(img):
    # 随机亮度
    value = random.uniform(0.5, 1.5)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * value, 0, 255)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # 随机翻转
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
        
    # 随机微小旋转
    rows, cols, _ = img.shape
    angle = random.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img = cv2.warpAffine(img, M, (cols, rows))
    
    return img

# 1. 获取你准备的原始图
raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith(('.jpg', '.png'))]
if not raw_files:
    print(f"错误：请先在 {RAW_DIR} 放几张参考图片！")
    exit()

dataset_entries = []
target_count = 1000
per_img_count = target_count // len(raw_files)

print(f">>> 正在根据 {len(raw_files)} 张原始图生成 {target_count} 张训练样本...")

count = 0
for raw_file in raw_files:
    img_path = os.path.join(RAW_DIR, raw_file)
    original_img = cv2.imread(img_path)
    
    # 假设文件名包含违章类型，比如 "solid_line_1.jpg"
    # 我们这里简单处理，你可以根据文件名手动定义逻辑
    vio_type = "压实线违章" if "line" in raw_file.lower() else "违规停放"
    if "normal" in raw_file.lower(): vio_type = "None"
    
    action = "ALARM" if vio_type != "None" else "CRUISE"
    thought = f"画面巡检分析：观察到车辆处于道路标线关键区域，判定为{vio_type}。"

    for i in range(per_img_count):
        aug_img = augment_image(original_img)
        filename = f"aug_{count:04d}.jpg"
        cv2.imwrite(os.path.join(IMG_DIR, filename), aug_img)
        
        dataset_entries.append({
            "messages": [
                {"from": "human", "value": "<image>\n作为智慧交通交警，仔细分析此画面。"},
                {"from": "gpt", "value": json.dumps({
                    "thought": thought,
                    "violation": vio_type,
                    "action": action
                }, ensure_ascii=False)}
            ],
            "images": [f"images/{filename}"]
        })
        count += 1
        if count % 100 == 0: print(f"  已生成 {count} 张...")

# 2. 写入 JSON
with open(os.path.join(BASE_DIR, "traffic_vla.json"), "w", encoding="utf-8") as f:
    json.dump(dataset_entries, f, ensure_ascii=False, indent=2)

print(f"\n>>> 🎉 成功！1000张增强数据集已准备就绪！")