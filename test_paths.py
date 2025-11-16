#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test path configuration for multicam_test_rgb.py
"""
import sys
import io
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 自動獲取項目根目錄
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
OPENNI2_BASE = PROJECT_ROOT / "NTK_CAP" / "ThirdParty" / "OpenNI2"

CAM_IPS = [
    "192.168.0.100",
    "192.168.3.100",
    "192.168.5.100",
    "192.168.7.100",
]

print("=" * 60)
print("路徑配置測試")
print("=" * 60)
print(f"腳本目錄: {SCRIPT_DIR}")
print(f"項目根目錄: {PROJECT_ROOT}")
print(f"OpenNI2 基礎路徑: {OPENNI2_BASE}")
print(f"OpenNI2 基礎路徑存在: {OPENNI2_BASE.exists()}")
print()

print("相機 OpenNI2 路徑檢查:")
print("-" * 60)
for ip in CAM_IPS:
    cam_path = OPENNI2_BASE / ip
    exists = cam_path.exists()
    status = "✓" if exists else "✗"
    print(f"{status} {ip:20s} -> {cam_path}")
    if exists:
        # 檢查關鍵文件
        dll_file = cam_path / "OpenNI2.dll"
        ini_file = cam_path / "OpenNI.ini"
        print(f"   - OpenNI2.dll: {'✓' if dll_file.exists() else '✗'}")
        print(f"   - OpenNI.ini:  {'✓' if ini_file.exists() else '✗'}")

print()
print("=" * 60)
print("路徑配置正確！可以運行 multicam_test_rgb.py")
print("=" * 60)
