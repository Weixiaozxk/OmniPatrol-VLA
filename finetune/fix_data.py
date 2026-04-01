import json
import os

DATA_DIR = r"E:\OmniPatrol-VLA\finetune\data"

# 修正为 LLaMA-Factory 标准的 from / value 格式
dataset = [
    {
        "messages":[
            {"from": "human", "value": "<image>作为智慧交通交警，分析此画面。"},
            {"from": "gpt", "value": "{\"thought\": \"画面中车辆压过白色实线，违反禁止标线指示。\", \"violation\": \"压实线违章\", \"action\": \"ALARM\"}"}
        ],
        "images": ["images/sample_1.jpg"]
    },
    {
        "messages":[
            {"from": "human", "value": "<image>作为智慧交通交警，分析此画面。"},
            {"from": "gpt", "value": "{\"thought\": \"画面中车辆压过黄色实线，存在逆行或违规超车风险。\", \"violation\": \"压黄实线违章\", \"action\": \"ALARM\"}"}
        ],
        "images":["images/sample_2.jpg"]
    },
    {
        "messages":[
            {"from": "human", "value": "<image>作为智慧交通交警，分析此画面。"},
            {"from": "gpt", "value": "{\"thought\": \"画面中车辆在车道内正常行驶，未触碰任何实线。\", \"violation\": \"None\", \"action\": \"CRUISE\"}"}
        ],
        "images": ["images/sample_3.jpg"]
    }
]

# 覆盖之前报错的 JSON 文件
with open(os.path.join(DATA_DIR, "traffic_vla.json"), "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(">>> 数据格式修复成功！请重新运行训练命令。")