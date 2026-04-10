# face_pose_bags_detect

**一个本地离线运行的检测系统:**  
<br>
😀 **人脸识别与认证** +  
<br>
🧍 **人体姿态检测** +   
<br>
🧳 **行李（背包/手提包/行李箱）检测**，  
<br>
支持`摄像头`与`本地视频文件`。
<br>

### 功能
- **人脸识别**：基于 `facenet-pytorch`提取特征，和本地人脸库做相似度匹配
- **姿态检测**：基于YOLO Pose
- **行李检测**：基于YOLO 检测
- **注册人脸**：从当前帧保存图片并写入数据库

### 目录结构（权重放置）
请按下面路径放好模型权重：

- **检测/姿态权重**：
  - `models/trained/best.pt`
  - `models/trained/pose.pt`
- **人脸权重（vggface2）**：
  - `models/facenet/20180402-114759-vggface2.pt`
  - `models/facenet/pnet.pt`
  - `models/facenet/rnet.pt`
  - `models/facenet/onet.pt`

### 安装与运行
建议使用 Python 3.11

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
python main.py
```

### 打包（PyInstaller）

```bash
pyinstaller --clean main.spec
```

输出目录：
- `dist/face_pose_bags_detect/`：可运行目录

