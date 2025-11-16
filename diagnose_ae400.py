#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AE400 相機診斷工具
檢查所有 AE400 相機的連接狀態
"""
import sys
import io
import os
import subprocess
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

def ping_camera(ip):
    """Ping 相機檢查網路連接"""
    try:
        # Windows: ping -n 1 -w 1000
        # Linux: ping -c 1 -W 1
        param = '-n' if sys.platform == 'win32' else '-c'
        timeout_param = '-w' if sys.platform == 'win32' else '-W'

        result = subprocess.run(
            ['ping', param, '1', timeout_param, '1000' if sys.platform == 'win32' else '1', ip],
            capture_output=True,
            text=True,
            timeout=3
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  Ping error: {e}")
        return False

def check_openni2_path(ip):
    """檢查 OpenNI2 路徑和配置"""
    cam_path = OPENNI2_BASE / ip
    if not cam_path.exists():
        return False, "路徑不存在"

    issues = []

    # 檢查必要文件
    dll_file = cam_path / "OpenNI2.dll"
    if not dll_file.exists():
        issues.append("缺少 OpenNI2.dll")

    ini_file = cam_path / "OpenNI.ini"
    if not ini_file.exists():
        issues.append("缺少 OpenNI.ini")

    network_json = cam_path / "OpenNI2" / "Drivers" / "network.json"
    if not network_json.exists():
        issues.append("缺少 network.json")
    else:
        # 檢查 network.json 配置
        try:
            import json
            with open(network_json, 'r') as f:
                config = json.load(f)

            ip1 = config.get('config', {}).get('ip1', '')
            if ip1 != ip:
                issues.append(f"network.json ip1={ip1} 不匹配 (應為 {ip})")
        except Exception as e:
            issues.append(f"network.json 讀取錯誤: {e}")

    if issues:
        return True, "; ".join(issues)
    return True, "配置正常"

def test_openni2_connection(ip):
    """嘗試使用 OpenNI2 連接相機"""
    import time
    openni2_path = str(OPENNI2_BASE / ip)

    try:
        from openni import openni2

        # 初始化 OpenNI2
        print(f"    初始化路徑: {openni2_path}", flush=True)
        openni2.initialize(openni2_path)

        # 短暫延遲讓初始化完成
        time.sleep(0.5)

        # 嘗試開啟設備
        dev = openni2.Device.open_any()

        # 獲取設備信息
        device_info = dev.get_device_info()

        # 關閉設備
        dev.close()
        time.sleep(0.2)  # 讓設備完全關閉

        openni2.unload()
        time.sleep(1.0)  # 重要！讓 unload 完全釋放資源

        return True, f"連接成功 - {device_info.name}"

    except Exception as e:
        try:
            openni2.unload()
            time.sleep(1.0)  # 即使失敗也要等待釋放
        except:
            pass
        return False, str(e)

def main():
    print("=" * 70)
    print("AE400 相機診斷工具")
    print("=" * 70)
    print(f"項目根目錄: {PROJECT_ROOT}")
    print(f"OpenNI2 基礎路徑: {OPENNI2_BASE}")
    print()

    results = []

    for i, ip in enumerate(CAM_IPS, 1):
        print(f"\n[{i}/4] 檢查相機: {ip}")
        print("-" * 70)

        # 1. Ping 測試
        print(f"  1. 網路連接測試...", end=" ", flush=True)
        can_ping = ping_camera(ip)
        if can_ping:
            print("✓ 可以 ping 通")
        else:
            print("✗ 無法 ping 通")

        # 2. 路徑檢查
        print(f"  2. OpenNI2 路徑檢查...", end=" ", flush=True)
        path_exists, path_msg = check_openni2_path(ip)
        if path_exists and "配置正常" in path_msg:
            print(f"✓ {path_msg}")
        elif path_exists:
            print(f"⚠ {path_msg}")
        else:
            print(f"✗ {path_msg}")

        # 3. OpenNI2 連接測試
        print(f"  3. OpenNI2 連接測試...", end=" ", flush=True)
        can_connect, connect_msg = test_openni2_connection(ip)
        if can_connect:
            print(f"✓ {connect_msg}")
        else:
            print(f"✗ {connect_msg}")

        # 記錄結果
        results.append({
            'ip': ip,
            'ping': can_ping,
            'path': path_exists and "配置正常" in path_msg,
            'connect': can_connect,
            'msg': connect_msg if not can_connect else "OK"
        })

        # 在測試之間增加延遲，確保資源完全釋放
        if i < len(CAM_IPS):
            import time
            print(f"\n  ⏳ 等待 2 秒讓資源完全釋放...\n", flush=True)
            time.sleep(2.0)

    # 總結
    print("\n" + "=" * 70)
    print("診斷總結")
    print("=" * 70)

    for r in results:
        status = "✓" if r['ping'] and r['path'] and r['connect'] else "✗"
        print(f"{status} {r['ip']:20s} - ", end="")
        if r['connect']:
            print("正常")
        else:
            if not r['ping']:
                print("網路不通")
            elif not r['path']:
                print("路徑/配置問題")
            else:
                print(f"OpenNI2 錯誤: {r['msg']}")

    # 建議
    failed_ips = [r['ip'] for r in results if not r['connect']]
    if failed_ips:
        print("\n" + "=" * 70)
        print("建議解決方案")
        print("=" * 70)
        for ip in failed_ips:
            r = next(r for r in results if r['ip'] == ip)
            print(f"\n{ip}:")
            if not r['ping']:
                print("  - 檢查相機是否開機")
                print("  - 檢查網路連接（網線、交換機）")
                print(f"  - 確認相機 IP 設定為 {ip}")
                print("  - 檢查電腦網路介面卡設定")
            elif not r['path']:
                print("  - 檢查 OpenNI2 資料夾是否完整")
                print(f"  - 確認路徑: {OPENNI2_BASE / ip}")
            else:
                print("  - 相機可能已被其他程式佔用")
                print("  - 嘗試重啟相機")
                print("  - 檢查 network.json 配置是否正確")
    else:
        print("\n✓ 所有相機連接正常！")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
