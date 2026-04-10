import os
import sys
import cv2
import numpy as np
import sqlite3
from datetime import datetime
from PIL import Image
import torch
from ultralytics import YOLO
from facenet_pytorch import MTCNN, InceptionResnetV1


def resource_path(rel_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, rel_path)
    return os.path.join(os.path.abspath("."), rel_path)


KNOWN_FACES_DIR = resource_path("known_faces")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

known_embeddings, known_names = [], []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ 使用设备: {device} ({'GPU加速' if device.type == 'cuda' else 'CPU'})")

yolo_model = YOLO(resource_path(os.path.join('models', 'best.pt')))
pose_model = YOLO(resource_path(os.path.join('models', 'pose.pt')))
yolo_model.to(device)
pose_model.to(device)

mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

threshold = 0.9

conn = sqlite3.connect(resource_path("db.sqlite3"))
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    image TEXT NOT NULL,
    created_at TEXT NOT NULL
)''')
conn.commit()


def close_db():
    conn.close()


def load_known_faces():
    embeddings = []
    names = []
    if not os.path.isdir(KNOWN_FACES_DIR):
        return embeddings, names
    for file in os.listdir(KNOWN_FACES_DIR):
        img_path = os.path.join(KNOWN_FACES_DIR, file)
        try:
            img = Image.open(img_path)
        except Exception:
            continue
        faces = mtcnn(img)
        if faces is not None and len(faces) > 0:
            embedding = resnet(faces[0].unsqueeze(0).to(device)).detach().cpu().numpy()[0]
            embeddings.append(embedding)
            name = os.path.splitext(file)[0].split('_')[0]
            names.append(name)
    return embeddings, names


def reload_faces():
    global known_embeddings, known_names
    known_embeddings, known_names = load_known_faces()


def validate_register_frame(frame):
    if frame is None:
        return "还没有视频帧可用于注册"
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    faces = mtcnn(img_pil)
    if faces is None or len(faces) == 0:
        return "未检测到人脸"
    return None


def register_new_face(frame, name):
    filename = f"{name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    path = os.path.join(KNOWN_FACES_DIR, filename)
    cv2.imwrite(path, frame)
    cursor.execute(
        "INSERT INTO users (name, image, created_at) VALUES (?, ?, ?)",
        (name, filename, datetime.now().isoformat()),
    )
    conn.commit()


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

                coco_skeleton = [
                    (5, 7), (7, 9), (6, 8), (8, 10),
                    (5, 6), (5, 11), (6, 12),
                    (11, 13), (13, 15), (12, 14), (14, 16), (11, 12)
                ]
                for start, end in coco_skeleton:
                    try:
                        x1, y1 = kpts[start]
                        x2, y2 = kpts[end]
                        if all(val > 0 for val in [x1, y1, x2, y2]):
                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    except Exception:
                        continue
    return frame


def recognize_and_draw_faces(frame):
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
    return frame
