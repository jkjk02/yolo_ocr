# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置 - YOLO OCR 尺寸识别器
使用方式: pyinstaller app/yolo_ocr.spec
"""

import os
import sys
from pathlib import Path

# 项目根目录
project_root = Path(SPECPATH).parent

block_cipher = None

a = Analysis(
    [str(project_root / 'app' / 'yolo_ocr.py')],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        # ONNX 模型
        (str(project_root / 'best-v2.onnx'), '.'),
        (str(project_root / 'model_b' / 'weights' / 'best.onnx'), 'model_b/weights'),
        # OCR 模型
        (str(project_root / 'onnxocr' / 'models'), 'onnxocr/models'),
        (str(project_root / 'onnxocr' / 'fonts'), 'onnxocr/fonts'),
    ],
    hiddenimports=[
        'ultralytics',
        'ultralytics.nn',
        'ultralytics.nn.tasks',
        'ultralytics.utils',
        'ultralytics.engine',
        'ultralytics.engine.results',
        'onnxruntime',
        'cv2',
        'numpy',
        'PIL',
        'PIL.Image',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        'IPython',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='yolo_ocr',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='yolo_ocr',
)
