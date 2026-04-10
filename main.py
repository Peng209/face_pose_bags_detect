import cv2
import os
import numpy as np
import sqlite3
from tkinter import simpledialog, messagebox, filedialog
from tkinter import Button, Frame, Label, Tk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from datetime import datetime
from threading import Thread
from ultralytics import YOLO
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import sys


# ---------- 路径处理 ----------
def resource_path(rel_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, rel_path)
    return os.path.join(os.path.abspath("."), rel_path)


# ---------- 全局变量 ----------
KNOWN_FACES_DIR = resource_path("known_faces")
known_embeddings, known_names = [], []
last_frame = None
running = False
current_source = 0  # 0: camera, 1: video file
video_cap = None


# ---------- 设备/模型加载 ----------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ 使用设备: {device} ({'GPU加速' if device.type == 'cuda' else 'CPU'})")

yolo_model = YOLO(resource_path('best.pt'))
pose_model = YOLO(resource_path('pose.pt'))
yolo_model.to(device)
pose_model.to(device)

# MTCNN
#  下载到了venv里
mtcnn = MTCNN(keep_all=True, device=device)

# InceptionResnetV1 - 会自动缓存模型到 ~/.cache/torch/checkpoints
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

threshold = 0.9


# ---------- 数据库 ----------
conn = sqlite3.connect(resource_path("db.sqlite3"))
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    image TEXT NOT NULL,
    created_at TEXT NOT NULL
)''')
conn.commit()


# ---------- 加载人脸库 ----------
def load_known_faces():
    known_embeddings = []
    known_names = []
    for file in os.listdir(KNOWN_FACES_DIR):
        img_path = os.path.join(KNOWN_FACES_DIR, file)
        try:
            img = Image.open(img_path)
        except Exception:
            continue
        faces = mtcnn(img)
        if faces is not None and len(faces) > 0:
            embedding = resnet(faces[0].unsqueeze(0).to(device)).detach().cpu().numpy()[0]
            known_embeddings.append(embedding)
            name = os.path.splitext(file)[0].split('_')[0]
            known_names.append(name)
    return known_embeddings, known_names


def reload_faces():
    global known_embeddings, known_names
    known_embeddings, known_names = load_known_faces()


# ---------- 注册人脸 ----------
def register_face(frame):
    if frame is None:
        messagebox.showerror("错误", "还没有视频帧可用于注册")
        return

    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    faces = mtcnn(img_pil)
    if faces is None or len(faces) == 0:
        messagebox.showerror("注册失败", "未检测到人脸")
        return

    name = simpledialog.askstring("注册人脸", "请输入用户名：")
    if not name:
        return
    filename = f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    path = os.path.join(KNOWN_FACES_DIR, filename)
    cv2.imwrite(path, frame)

    cursor.execute("INSERT INTO users (name, image, created_at) VALUES (?, ?, ?)",
                   (name, filename, datetime.now().isoformat()))
    conn.commit()
    messagebox.showinfo("注册成功", f"用户 {name} 已注册")
    reload_faces()


# ---------- 检测行李 + 姿态 ----------
def detect_all(frame):
    detect_results = yolo_model(frame, verbose=False)[0]
    for box in detect_results.boxes:
        cls = int(box.cls[0])
        name = yolo_model.model.names[cls]
        if name in ['suitcase', 'handbag', 'backpack']:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 200, 50), 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 200, 50), 2)

    pose_results = pose_model(frame, verbose=False)[0]
    if pose_results.keypoints is not None and pose_results.keypoints.xy is not None:
        kpts_batch = pose_results.keypoints.xy
        if kpts_batch.ndim == 3:
            for kpts in kpts_batch:
                if kpts.ndim != 2 or kpts.shape[0] != 17:
                    continue
                for i, kp in enumerate(kpts):
                    x, y = kp
                    if x > 0 and y > 0:
                        if i in [6, 7, 9, 10]:
                            radius = 6
                        elif i in [0, 1, 2, 3, 4]:
                            continue
                        else:
                            radius = 4
                        cv2.circle(frame, (int(x), int(y)), radius, (0, 255, 0), -1)

                COCO_SKELETON = [
                    (5, 7), (7, 9), (6, 8), (8, 10),
                    (5, 6), (5, 11), (6, 12),
                    (11, 13), (13, 15), (12, 14), (14, 16), (11, 12)
                ]
                for start, end in COCO_SKELETON:
                    try:
                        x1, y1 = kpts[start]
                        x2, y2 = kpts[end]
                        if all(val > 0 for val in [x1, y1, x2, y2]):
                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    except Exception:
                        continue
    return frame


# ---------- 视频流逻辑 ----------
def select_video_file():
    filepath = filedialog.askopenfilename(
        title="选择视频文件",
        filetypes=[("视频文件", "*.mp4 *.avi *.mov"), ("所有文件", "*.*")]
    )
    return filepath if filepath else None


def start_video(source_type):
    global running, current_source, video_cap
    running = True
    current_source = source_type

    if source_type == 1:
        video_path = select_video_file()
        if not video_path:
            return
        video_cap = cv2.VideoCapture(video_path)
    else:
        video_cap = cv2.VideoCapture(0)

    Thread(target=video_loop, daemon=True).start()


def stop_video():
    global running, video_cap
    running = False
    if video_cap is not None:
        video_cap.release()
        video_cap = None


def video_loop():
    global video_cap, last_frame
    while running and video_cap is not None:
        ret, frame = video_cap.read()
        if not ret:
            if current_source == 1:
                stop_video()
                messagebox.showinfo("提示", "视频播放结束")
            break

        last_frame = frame.copy()  # ✅ 实时更新用于注册
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        faces_tensor = mtcnn(img_pil)
        boxes, _ = mtcnn.detect(img_pil)

        if faces_tensor is not None and boxes is not None:
            for face_tensor, box in zip(faces_tensor, boxes):
                x1, y1, x2, y2 = map(int, box)
                embedding = resnet(face_tensor.unsqueeze(0).to(device)).detach().cpu().numpy()[0]
                name = "Unknown"
                if known_embeddings:
                    distances = [np.linalg.norm(embedding - emb) for emb in known_embeddings]
                    min_index = np.argmin(distances)
                    if distances[min_index] < threshold:
                        name = known_names[min_index]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        frame = detect_all(frame)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        video_label.config(image=img)
        video_label.image = img


# ---------- GUI ----------
def on_exit():
    stop_video()
    conn.close()
    root.quit()


root = Tk()
root.title("人脸识别 + 行李识别系统")
root.geometry("1080x720")

video_label = Label(root)
video_label.pack()

btn_frame = Frame(root)
btn_frame.pack(pady=10)

Button(btn_frame, text="摄像头识别", command=lambda: start_video(0), width=15).grid(row=0, column=0, padx=10)
Button(btn_frame, text="视频文件识别", command=lambda: start_video(1), width=15).grid(row=0, column=1, padx=10)
Button(btn_frame, text="停止识别", command=stop_video, width=15).grid(row=0, column=2, padx=10)
Button(btn_frame, text="注册人脸", command=lambda: register_face(last_frame), width=15).grid(row=1, column=0, padx=10, pady=10)
Button(btn_frame, text="重新加载人脸", command=reload_faces, width=15).grid(row=1, column=1, padx=10, pady=10)
Button(btn_frame, text="退出", command=on_exit, width=15).grid(row=1, column=2, padx=10, pady=10)

reload_faces()
root.mainloop()
