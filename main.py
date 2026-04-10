from tkinter import Button, Frame, Label, Tk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from threading import Thread
import cv2

from service import logic


last_frame = None
running = False
current_source = 0  # 0: camera, 1: video file
video_cap = None


def select_video_file():
    filepath = filedialog.askopenfilename(
        title="选择视频文件",
        filetypes=[("视频文件", "*.mp4 *.avi *.mov"), ("所有文件", "*.*")],
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

        last_frame = frame.copy()
        frame = logic.recognize_and_draw_faces(frame)
        frame = logic.detect_all(frame)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img))
        video_label.config(image=img)
        video_label.image = img


def register_face(frame):
    err = logic.validate_register_frame(frame)
    if err:
        messagebox.showerror("错误" if "帧" in err else "注册失败", err)
        return

    name = simpledialog.askstring("注册人脸", "请输入用户名：")
    if not name:
        return
    logic.register_new_face(frame, name)
    messagebox.showinfo("注册成功", f"用户 {name} 已注册")
    logic.reload_faces()


def on_exit():
    stop_video()
    logic.close_db()
    root.quit()


root = Tk()
root.title("人脸认证 + 姿态检测 + 行李识别系统")
root.geometry("1080x720")

video_label = Label(root)
video_label.pack()

btn_frame = Frame(root)
btn_frame.pack(pady=10)

Button(btn_frame, text="摄像头识别", command=lambda: start_video(0), width=15).grid(
    row=0, column=0, padx=10
)
Button(btn_frame, text="视频文件识别", command=lambda: start_video(1), width=15).grid(
    row=0, column=1, padx=10
)
Button(btn_frame, text="停止识别", command=stop_video, width=15).grid(
    row=0, column=2, padx=10
)
Button(
    btn_frame, text="注册人脸", command=lambda: register_face(last_frame), width=15
).grid(row=1, column=0, padx=10, pady=10)
Button(btn_frame, text="重新加载人脸", command=logic.reload_faces, width=15).grid(
    row=1, column=1, padx=10, pady=10
)
Button(btn_frame, text="退出", command=on_exit, width=15).grid(
    row=1, column=2, padx=10, pady=10
)

logic.reload_faces()
root.mainloop()
