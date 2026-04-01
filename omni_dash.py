import sys
import os
import cv2
import requests
import threading
import time
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QTextEdit, QPushButton, QFrame, QGridLayout)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject

# ================= 后台 VLA 推理通信线程 =================
class VLAWorker(QObject):
    # 定义信号，用于将子线程的推理结果安全地传给主界面
    result_ready = pyqtSignal(dict, object)
    
    def __init__(self):
        super().__init__()
        self.running = False
        
    def patrol_loop(self, get_frame_cb, url):
        self.running = True
        while self.running:
            frame = get_frame_cb()
            if frame is not None:
                try:
                    # 将 OpenCV 画面编码发送给 4060 后端
                    _, img_encoded = cv2.imencode('.jpg', frame)
                    files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}
                    res = requests.post(url, files=files, timeout=5)
                    if res.status_code == 200:
                        # 触发信号，携带 JSON 结果和触发时的画面
                        self.result_ready.emit(res.json(), frame)
                except Exception as e:
                    print(f"通信异常: {e}")
                    pass
            # 设定巡检频率：每 3 秒抽帧送去分析，保护显卡
            time.sleep(3)

# ================= 主界面 UI 及控制逻辑 =================
class OmniPatrolPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OmniPatrol-VLA 智慧交通多模态巡检终端 - 独立开发旗舰版")
        self.setFixedSize(1280, 800)
        
        self.cap = None
        self.current_frame = None
        self.host_url = "http://127.0.0.1:8001/analyze"  # 联调 RDK X5 时改为你主机的局域网 IP
        
        # 抓拍图片保存路径
        self.capture_dir = "E:/OmniPatrol-VLA/Violations_DB"
        os.makedirs(self.capture_dir, exist_ok=True)
        
        self.initUI()
        
        # 视频刷新定时器 (30ms 刷新一次界面，保证 30帧+)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # VLA 后台线程初始化
        self.worker = VLAWorker()
        self.worker.result_ready.connect(self.handle_vla_result)
        self.vla_thread = threading.Thread(target=self.worker.patrol_loop, args=(self.get_current_frame, self.host_url), daemon=True)

    def initUI(self):
        # --- 赛博朋克深色极客主题 ---
        self.setStyleSheet("""
            QMainWindow { background-color: #0d1117; }
            QLabel { color: #c9d1d9; font-family: 'Segoe UI'; font-size: 14px; }
            QFrame { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; }
            QPushButton { background-color: #238636; color: white; border-radius: 6px; font-weight: bold; font-size: 16px; border: none; }
            QPushButton:hover { background-color: #2ea043; }
            QPushButton#stopBtn { background-color: #da3633; }
            QTextEdit { background-color: #010409; color: #58a6ff; border: 1px solid #30363d; border-radius: 5px; font-family: Consolas; font-size: 14px; padding: 5px; }
        """)
        
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)
        
        # ================= 左侧：视觉流与机器人底盘参数 =================
        left_panel = QVBoxLayout()
        
        # 1. 主视觉大屏
        self.video_label = QLabel("NO CAMERA SIGNAL\n等待视觉接入")
        self.video_label.setFixedSize(800, 600)
        self.video_label.setStyleSheet("background-color: #000; color: #555; font-size: 24px; border: 2px solid #58a6ff;")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_panel.addWidget(self.video_label)
        
        # 2. 机器人实时参数仪表盘 (未来可接 ROS2 下位机回传数据)
        dash_frame = QFrame()
        dash_frame.setFixedHeight(80)
        dash_layout = QGridLayout(dash_frame)
        self.lbl_speed = QLabel("🚀 实时速度: 0.0 m/s")
        self.lbl_bat = QLabel("🔋 电池电量: 98%")
        self.lbl_mode = QLabel("📡 导航模式: Nav2 自动巡航")
        self.lbl_gps = QLabel("📍 定位状态: RTK Fixed")
        dash_layout.addWidget(self.lbl_speed, 0, 0)
        dash_layout.addWidget(self.lbl_bat, 0, 1)
        dash_layout.addWidget(self.lbl_mode, 1, 0)
        dash_layout.addWidget(self.lbl_gps, 1, 1)
        left_panel.addWidget(dash_frame)
        
        # 3. 硬件控制按键
        btn_layout = QHBoxLayout()
        self.btn_cam = QPushButton("📷 连接机器人主视角")
        self.btn_cam.setFixedHeight(50)
        self.btn_cam.clicked.connect(self.toggle_cam)
        
        self.btn_auto = QPushButton("▶ 启动全自动 VLA 巡检")
        self.btn_auto.setFixedHeight(50)
        self.btn_auto.clicked.connect(self.toggle_patrol)
        
        btn_layout.addWidget(self.btn_cam)
        btn_layout.addWidget(self.btn_auto)
        left_panel.addLayout(btn_layout)
        
        layout.addLayout(left_panel, 7)
        
        # ================= 右侧：VLA 核心与抓拍记录 =================
        right_panel = QVBoxLayout()
        
        # 1. 警报核心区
        self.alert_frame = QFrame()
        self.alert_frame.setFixedHeight(100)
        alert_layout = QVBoxLayout(self.alert_frame)
        self.lbl_alert_title = QLabel("SYSTEM STATUS: OK")
        self.lbl_alert_title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.lbl_alert_title.setStyleSheet("color: #3fb950;")
        self.lbl_alert_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        alert_layout.addWidget(self.lbl_alert_title)
        right_panel.addWidget(self.alert_frame)
        
        # 2. VLA 思维流透视 (显示大模型 thought)
        right_panel.addWidget(QLabel("🧠 VLA 实时视觉理解 (Thought):"))
        self.thought_box = QTextEdit()
        self.thought_box.setFixedHeight(120)
        right_panel.addWidget(self.thought_box)
        
        # 3. 自动取证日志
        right_panel.addWidget(QLabel("📂 自动取证抓拍记录 (违章库):"))
        self.log_box = QTextEdit()
        self.log_box.setStyleSheet("color: #8b949e; background-color: #010409;")
        self.log_box.setReadOnly(True)
        right_panel.addWidget(self.log_box)
        
        layout.addLayout(right_panel, 3)
        self.setCentralWidget(main_widget)

    # --- 功能实现区 ---
    def toggle_cam(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
            self.btn_cam.setText("🛑 断开主视角")
        else:
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.video_label.clear()
            self.video_label.setText("NO SIGNAL")
            self.lbl_speed.setText("🚀 实时速度: 0.0 m/s")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            # BGR 转 RGB 给 PyQt 显示
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            q_img = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_img).scaled(800, 600, Qt.AspectRatioMode.KeepAspectRatio))
            
            # 动态模拟底盘速度 (巡检状态下有速度，待机为0)
            if self.worker.running:
                self.lbl_speed.setText(f"🚀 实时速度: 1.{int(time.time()*10)%9} m/s")
            else:
                self.lbl_speed.setText("🚀 实时速度: 0.0 m/s")

    def get_current_frame(self):
        return self.current_frame

    def toggle_patrol(self):
        if not self.worker.running:
            if self.cap is None:
                self.log_box.append("[系统警告] 请先连接机器人视角！")
                return
            # UI 状态切换为运行
            self.btn_auto.setText("⏹ 停止自动巡检")
            self.btn_auto.setObjectName("stopBtn")
            self.btn_auto.style().unpolish(self.btn_auto)
            self.btn_auto.style().polish(self.btn_auto)
            
            if not self.vla_thread.is_alive():
                self.vla_thread = threading.Thread(target=self.worker.patrol_loop, args=(self.get_current_frame, self.host_url), daemon=True)
                self.vla_thread.start()
            else:
                self.worker.running = True
        else:
            # UI 状态切换为停止
            self.worker.running = False
            self.btn_auto.setText("▶ 启动全自动 VLA 巡检")
            self.btn_auto.setObjectName("")
            self.btn_auto.style().unpolish(self.btn_auto)
            self.btn_auto.style().polish(self.btn_auto)
            self.lbl_speed.setText("🚀 实时速度: 0.0 m/s")

    def handle_vla_result(self, data, frame):
        """处理从后端传回来的分析结果，并更新 UI"""
        thought = data.get("thought", "")
        action = data.get("action", "CRUISE")

        self.thought_box.setText(thought)

        if action == "ALARM":
            # 1. 界面爆红报警
            self.alert_frame.setStyleSheet("background-color: #4a0000; border: 2px solid #ff0000;")
            self.lbl_alert_title.setStyleSheet("color: #ff4444;")
            self.lbl_alert_title.setText("🔥 VIOLATION DETECTED!")
            
            # 2. 自动取证抓拍保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{self.capture_dir}/vio_{timestamp}.jpg"
            cv2.imwrite(save_path, frame)
            
            # 3. 打印日志
            log_msg = f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ 发现违章，自动抓拍取证！\n ↳ 路径: {save_path}\n"
            self.log_box.append(log_msg)
            
            # 模拟机器人减速停车
            self.lbl_speed.setText("🚀 实时速度: 0.0 m/s (停车取证)")
        else:
            # 恢复正常绿灯状态
            self.alert_frame.setStyleSheet("background-color: #161b22;")
            self.lbl_alert_title.setStyleSheet("color: #3fb950;")
            self.lbl_alert_title.setText("SYSTEM STATUS: OK")


# ================= 保证程序能够正常启动的入口点 =================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OmniPatrolPro()
    window.show()
    sys.exit(app.exec())