# -*- coding: utf-8 -*-
"""
簡單四攝影機顯示程式
使用標準 OpenCV（不需要 OpenGL）
"""

import cv2
import numpy as np
import time

class SimpleCameraDisplay:
    def __init__(self):
        self.cameras = []
        self.window_name = "四攝影機顯示"
        self.use_test_pattern = False
        
        # 統計資訊
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.fps = 0.0
        
        # 初始化攝影機
        self.init_cameras()
    
    def init_cameras(self):
        """初始化四個攝影機"""
        print("正在初始化攝影機...")
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                self.cameras.append(cap)
                print(f"✅ 攝影機 {i} 初始化成功")
            else:
                print(f"❌ 攝影機 {i} 不可用")
                self.cameras.append(None)
        
        # 檢查是否需要測試模式
        available_cameras = [i for i, cam in enumerate(self.cameras) if cam is not None]
        if not available_cameras:
            print("沒有偵測到攝影機，使用測試畫面")
            self.use_test_pattern = True
        else:
            print(f"找到 {len(available_cameras)} 個可用攝影機: {available_cameras}")
    
    def create_test_frame(self, camera_id):
        """創建測試畫面"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        current_time = time.time()
        wave = int(127 * (1 + np.sin(current_time * 2 + camera_id)))
        
        # 不同顏色
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        color = colors[camera_id % 4]
        
        # 填充背景色
        frame[:, :] = [c * wave // 255 for c in color]
        
        # 添加文字
        cv2.putText(frame, f"Test Camera {camera_id}", (180, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Time: {current_time:.1f}", (200, 260), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 添加移動圓形
        center_x = int(320 + 100 * np.sin(current_time + camera_id))
        center_y = int(240 + 50 * np.cos(current_time * 1.5 + camera_id))
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), 2)
        
        return frame
    
    def get_camera_frames(self):
        """獲取所有攝影機畫面"""
        frames = []
        
        for i in range(4):
            if self.use_test_pattern:
                frame = self.create_test_frame(i)
                frames.append(frame)
            elif i < len(self.cameras) and self.cameras[i] is not None:
                ret, frame = self.cameras[i].read()
                if ret:
                    frames.append(frame)
                else:
                    # 讀取失敗
                    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(error_frame, f"Camera {i} Error", 
                               (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    frames.append(error_frame)
            else:
                # 攝影機不可用
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"Camera {i} N/A", 
                           (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
                frames.append(placeholder)
        
        return frames
    
    def add_overlay_info(self, frame, camera_id):
        """在畫面上添加覆蓋資訊"""
        # 標題列
        label_bg = np.zeros((40, 640, 3), dtype=np.uint8)
        label_bg[:] = (50, 50, 50)
        frame[0:40, :] = label_bg
        
        cv2.putText(frame, f"Camera {camera_id}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 時間戳記
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(frame, timestamp, (450, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        return frame
    
    def create_grid_display(self, frames):
        """創建 2x2 網格顯示"""
        # 確保有四個畫面
        while len(frames) < 4:
            frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
        
        # 處理每個畫面
        processed_frames = []
        for i, frame in enumerate(frames[:4]):
            # 調整大小
            resized = cv2.resize(frame, (640, 480))
            
            # 添加標籤
            processed = self.add_overlay_info(resized, i)
            processed_frames.append(processed)
        
        # 組合 2x2 網格
        top_row = np.hstack((processed_frames[0], processed_frames[1]))
        bottom_row = np.hstack((processed_frames[2], processed_frames[3]))
        combined = np.vstack((top_row, bottom_row))
        
        # 添加邊框
        border_color = (100, 100, 100)
        cv2.rectangle(combined, (0, 0), (1279, 959), border_color, 2)
        cv2.line(combined, (640, 0), (640, 960), border_color, 2)  # 垂直線
        cv2.line(combined, (0, 480), (1280, 480), border_color, 2)  # 水平線
        
        # 添加標題欄
        title_bg = np.zeros((30, 1280, 3), dtype=np.uint8)
        title_bg[:] = (30, 30, 30)
        
        combined_with_title = np.vstack((title_bg, combined))
        
        # 標題文字
        cv2.putText(combined_with_title, "NTKCAP 四攝影機顯示系統", (400, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # FPS 資訊
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(combined_with_title, fps_text, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # 模式資訊
        mode_text = "Test Mode" if self.use_test_pattern else "Camera Mode"
        cv2.putText(combined_with_title, mode_text, (1100, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        return combined_with_title
    
    def calculate_fps(self):
        """計算 FPS"""
        current_time = time.time()
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = current_time
        
        self.frame_count += 1
    
    def run(self):
        """運行主程式"""
        print(f"\n四攝影機顯示程式運行中...")
        print(f"OpenCV 版本: {cv2.__version__}")
        print(f"測試模式: {'是' if self.use_test_pattern else '否'}")
        
        print("\n控制按鍵:")
        print("  'q' 或 ESC - 退出程式")
        print("  's' - 保存截圖")
        print("  'r' - 重新初始化攝影機")
        print("  'i' - 顯示系統資訊")
        print()
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        
        try:
            while True:
                # 獲取攝影機畫面
                frames = self.get_camera_frames()
                
                # 創建網格顯示
                display_frame = self.create_grid_display(frames)
                
                # 計算 FPS
                self.calculate_fps()
                
                # 顯示畫面
                cv2.imshow(self.window_name, display_frame)
                
                # 處理按鍵
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' 或 ESC
                    break
                elif key == ord('s'):  # 保存截圖
                    self.save_screenshot(display_frame)
                elif key == ord('r'):  # 重新初始化
                    self.reinit_cameras()
                elif key == ord('i'):  # 顯示系統資訊
                    self.show_system_info()
                
        except KeyboardInterrupt:
            print("\n程式被用戶中斷")
        except Exception as e:
            print(f"程式運行錯誤: {e}")
        finally:
            self.cleanup()
    
    def save_screenshot(self, frame):
        """保存截圖"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"4cameras_{timestamp}.png"
        
        try:
            cv2.imwrite(filename, frame)
            print(f"✅ 截圖已保存: {filename}")
        except Exception as e:
            print(f"❌ 保存截圖失敗: {e}")
    
    def show_system_info(self):
        """顯示系統資訊"""
        print("\n=== 系統資訊 ===")
        print(f"OpenCV 版本: {cv2.__version__}")
        print(f"測試模式: {self.use_test_pattern}")
        print(f"當前 FPS: {self.fps:.2f}")
        print(f"運行時間: {time.time() - self.start_time:.1f} 秒")
        available_cameras = [i for i, cam in enumerate(self.cameras) if cam is not None]
        print(f"可用攝影機: {available_cameras}")
        print("================\n")
    
    def reinit_cameras(self):
        """重新初始化攝影機"""
        print("重新初始化攝影機...")
        
        # 釋放現有攝影機
        for cam in self.cameras:
            if cam is not None:
                cam.release()
        
        # 重新初始化
        self.cameras = []
        self.init_cameras()
    
    def cleanup(self):
        """清理資源"""
        print("\n正在清理資源...")
        
        # 釋放攝影機
        for cam in self.cameras:
            if cam is not None:
                cam.release()
        
        # 關閉視窗
        cv2.destroyAllWindows()
        print("✅ 資源清理完成")


def main():
    """主函數"""
    print("NTKCAP 簡單四攝影機顯示程式")
    print("=" * 50)
    print(f"OpenCV 版本: {cv2.__version__}")
    
    try:
        display = SimpleCameraDisplay()
        display.run()
        
    except KeyboardInterrupt:
        print("\n程式被用戶中斷")
    except Exception as e:
        print(f"程式運行失敗: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()