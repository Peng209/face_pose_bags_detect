# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[os.path.abspath(".")],
    binaries=[],
    datas=[
        ('models/best.pt', 'models'),     # 模型文件
        ('models/pose.pt', 'models'),     # 模型文件
        ('db.sqlite3', '.'),                 # SQLite 数据库
        ('known_faces', 'known_faces'),     # 人脸图像文件夹
    ],
    hiddenimports=collect_submodules("ultralytics") + collect_submodules("facenet_pytorch"),
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='face_pose_bags_detect',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False  # 设置为 True 会弹出黑框，GUI 建议为 False
)
