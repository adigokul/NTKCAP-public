#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import cv2
from multiprocessing import Process
from openni import openni2
from pathlib import Path

# ====== 相機設定 ======
# 自動獲取項目根目錄
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR  # 腳本在項目根目錄
OPENNI2_BASE = PROJECT_ROOT / "NTK_CAP" / "ThirdParty" / "OpenNI2"

CAM_IPS = [
    "192.168.0.100",
    "192.168.3.100",
    "192.168.5.100",
    "192.168.7.100",
]

VIEW_W, VIEW_H = 640, 480


TIMEOUT_MS = 2000             
MISSES_BEFORE_RESET = 3         
RESETS_BEFORE_REOPEN = 2       
REOPENS_BEFORE_REINIT = 2       



def camera_worker(init_path, window_title):
   
    dev = None
    color_stream = None

    misses = 0
    resets_done = 0
    reopens_done = 0

    def start_everything():
        nonlocal dev, color_stream
        openni2.initialize(init_path)
        dev = openni2.Device.open_any()

        
        color_stream = dev.create_color_stream()
       
        color_stream.start()
        print(f"[{window_title}] initialized & color stream started.")

    def stop_stream():
        nonlocal color_stream
        try:
            if color_stream:
                color_stream.stop()
        except Exception:
            pass

    def restart_stream():
        nonlocal color_stream
        try:
            stop_stream()
            color_stream = dev.create_color_stream()
            color_stream.start()
            print(f"[{window_title}] color stream restarted.")
            return True
        except Exception as e:
            print(f"[{window_title}] color stream restart failed: {e}")
            return False

    def reopen_device():
        nonlocal dev, color_stream
        try:
            stop_stream()
            openni2.unload()
            openni2.initialize(init_path)
            dev = openni2.Device.open_any()
            color_stream = dev.create_color_stream()
            color_stream.start()
            print(f"[{window_title}] device reopened & color stream started.")
            return True
        except Exception as e:
            print(f"[{window_title}] device reopen failed: {e}")
            return False

    def reinitialize_all():
        try:
            stop_stream()
            openni2.unload()
        except Exception:
            pass
        start_everything()

    try:
        start_everything()
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

        while True:
            try:
                idx = openni2.wait_for_any_stream([color_stream], TIMEOUT_MS)
                if idx is None:
                    misses += 1
                    print(f"[{window_title}] timeout {misses}/{MISSES_BEFORE_RESET}")
                    if misses >= MISSES_BEFORE_RESET:
                        if restart_stream():
                            misses = 0
                            resets_done += 1
                            continue
                        else:
                            resets_done += 1
                            if resets_done >= RESETS_BEFORE_REOPEN:
                                if reopen_device():
                                    misses = 0
                                    resets_done = 0
                                    reopens_done += 1
                                    continue
                                else:
                                    if reopens_done >= REOPENS_BEFORE_REINIT:
                                        print(f"[{window_title}] reinit all due to repeated failures.")
                                        reinitialize_all()
                                        misses = 0
                                        resets_done = 0
                                        reopens_done = 0
                                        continue
                    continue

                # 正常讀取一幀 color（RGB888）
                frame = color_stream.read_frame()
                buf = frame.get_buffer_as_uint8()
                vm = color_stream.get_video_mode()
                w, h = vm.resolutionX, vm.resolutionY

                # 轉成 (H, W, 3) 的 RGB，再轉 BGR 給 OpenCV 顯示
                rgb = np.asarray(buf).reshape((h, w, 3))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                bgr = cv2.resize(bgr, (VIEW_W, VIEW_H), interpolation=cv2.INTER_LINEAR)

                cv2.imshow(window_title, bgr)

                
                misses = 0

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

            except Exception as e:
                print(f"[{window_title}] read error: {e}")
                misses += 1
                if misses >= MISSES_BEFORE_RESET:
                    if not restart_stream():
                        if not reopen_device():
                            print(f"[{window_title}] reinitializing after read errors.")
                            reinitialize_all()
                    misses = 0
                    resets_done = 0
                    reopens_done = 0

    except Exception as e:
        print(f"[{window_title}] fatal error: {e}")
        time.sleep(2)

    finally:
        try:
            stop_stream()
            openni2.unload()
            cv2.destroyWindow(window_title)
        except Exception:
            pass
        print(f"[{window_title}] closed.")


def main():
    # 為每台相機準備 initialize 路徑：OPENNI2_BASE/<IP>
    init_paths = [str(OPENNI2_BASE / ip) for ip in CAM_IPS]

    # 驗證路徑存在
    print(f"項目根目錄: {PROJECT_ROOT}")
    print(f"OpenNI2 基礎路徑: {OPENNI2_BASE}")
    for ip, init_path in zip(CAM_IPS, init_paths):
        if not Path(init_path).exists():
            print(f"警告: 路徑不存在 - {init_path}")
        else:
            print(f"✓ 找到相機 {ip} 的 OpenNI2 路徑")

    procs = []
    print("\n開始啟動相機...")
    for i, (ip, init_path) in enumerate(zip(CAM_IPS, init_paths)):
        title = f"RGB View - {ip}"
        print(f"[{i+1}/4] 正在啟動 {ip}...", flush=True)
        p = Process(target=camera_worker, args=(init_path, title), daemon=True)
        p.start()
        procs.append(p)
        time.sleep(1.5)  # 增加延遲避免併發初始化衝突

    print("\n四個 RGB 視窗已啟動。任何視窗按 q 或 ESC 可關閉。")
    print("若要整體退出，請關閉所有視窗或在此終端按 Ctrl+C。")

    try:
        while any(p.is_alive() for p in procs):
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("收到中斷，正在結束所有子行程...")
    finally:
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join()
        print("全部結束，再見！")


if __name__ == "__main__":
    main()
