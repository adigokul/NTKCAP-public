#!/usr/bin/env python3
"""
WebSocket EMG Data Stream Verification Script
驗證 ws://localhost:31278/ws 是否有EMG數據流
"""

import websocket
import json
import time
import threading
import sys

class EMGWebSocketTester:
    def __init__(self, uri="ws://localhost:31278/ws"):
        self.uri = uri
        self.ws = None
        self.connected = False
        self.data_count = 0
        self.last_data_time = None
        self.connection_established = False
        
    def on_message(self, ws, message):
        """處理接收到的WebSocket消息"""
        try:
            self.data_count += 1
            self.last_data_time = time.time()
            
            # 解析JSON數據
            data = json.loads(message)
            
            # 顯示數據信息
            if self.data_count == 1:
                print("✅ First data packet received!")
                print(f"📦 Data structure keys: {list(data.keys())}")
                
            if self.data_count <= 5 or self.data_count % 100 == 0:
                print(f"📊 Packet #{self.data_count}: {time.strftime('%H:%M:%S')}")
                if 'data' in data:
                    print(f"   Data length: {len(data['data']) if isinstance(data['data'], list) else 'Not a list'}")
                if 'timestamp' in data:
                    print(f"   Timestamp: {data['timestamp']}")
                    
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON decode error: {e}")
            print(f"   Raw message: {message[:100]}...")
        except Exception as e:
            print(f"❌ Error processing message: {e}")
    
    def on_error(self, ws, error):
        """處理WebSocket錯誤"""
        print(f"❌ WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        """處理WebSocket關閉"""
        print(f"🔌 Connection closed: Status={close_status_code}, Message={close_msg}")
        self.connected = False
    
    def on_open(self, ws):
        """處理WebSocket開啟"""
        print("✅ WebSocket connection established!")
        self.connected = True
        self.connection_established = True
    
    def test_connection(self, duration=10):
        """測試WebSocket連接和數據流"""
        print(f"🎯 Testing EMG WebSocket: {self.uri}")
        print(f"⏱️  Test duration: {duration} seconds")
        print("-" * 50)
        
        try:
            # 創建WebSocket連接
            websocket.enableTrace(True)  # 啟用詳細日誌
            self.ws = websocket.WebSocketApp(
                self.uri,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
                on_open=self.on_open
            )
            
            # 啟動WebSocket連接（在背景執行）
            def run_websocket():
                self.ws.run_forever()
            
            ws_thread = threading.Thread(target=run_websocket)
            ws_thread.daemon = True
            ws_thread.start()
            
            # 等待連接建立
            wait_time = 0
            while not self.connection_established and wait_time < 5:
                time.sleep(0.1)
                wait_time += 0.1
            
            if not self.connection_established:
                print("❌ Failed to establish connection within 5 seconds")
                return False
            
            # 監控數據流
            start_time = time.time()
            last_count = 0
            
            print(f"👂 Listening for EMG data...")
            
            while time.time() - start_time < duration:
                time.sleep(1)
                current_time = time.time()
                
                # 顯示統計信息
                if self.data_count > last_count:
                    rate = self.data_count - last_count
                    print(f"📈 Data rate: {rate} packets/sec (Total: {self.data_count})")
                    last_count = self.data_count
                elif self.data_count == 0:
                    print(f"⏳ Waiting for data... ({int(current_time - start_time)}s)")
                
                # 檢查是否仍然連接
                if not self.connected:
                    print("❌ Connection lost!")
                    break
            
            # 關閉連接
            if self.ws:
                self.ws.close()
            
            return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def print_summary(self):
        """顯示測試總結"""
        print("\n" + "=" * 50)
        print("📋 Test Summary:")
        print(f"   Connection established: {'✅ Yes' if self.connection_established else '❌ No'}")
        print(f"   Total data packets: {self.data_count}")
        print(f"   Last data received: {time.strftime('%H:%M:%S', time.localtime(self.last_data_time)) if self.last_data_time else 'None'}")
        
        if self.data_count > 0:
            print("✅ EMG WebSocket service is working and providing data!")
        elif self.connection_established:
            print("⚠️  Connection successful but no data received")
            print("   - Check if EMG device is connected")
            print("   - Check if Cygnus software is streaming data")
        else:
            print("❌ Cannot connect to EMG WebSocket service")
            print("   - Check if Cygnus software is running")
            print("   - Check if WebSocket server is on localhost:31278")

def test_alternative_ports():
    """測試其他可能的EMG端口"""
    print("\n🔍 Testing alternative ports...")
    
    alternative_uris = [
        "ws://localhost:31278",      # 沒有/ws後綴
        "ws://127.0.0.1:31278/ws",   # 使用127.0.0.1
        "ws://127.0.0.1:31278",      # 127.0.0.1 沒有/ws
        "ws://localhost:8080/ws",    # 常見的WebSocket端口
        "ws://localhost:9001/ws",    # 另一個常見端口
    ]
    
    for uri in alternative_uris:
        print(f"\n🔍 Testing: {uri}")
        tester = EMGWebSocketTester(uri)
        
        # 快速連接測試（2秒）
        try:
            ws = websocket.create_connection(uri, timeout=2)
            print(f"✅ Connection successful: {uri}")
            ws.close()
            
            # 如果連接成功，進行完整測試
            success = tester.test_connection(5)
            if success and tester.data_count > 0:
                print(f"🎯 Found working EMG service at: {uri}")
                return uri
                
        except Exception as e:
            print(f"❌ Failed: {str(e)}")
    
    return None

if __name__ == "__main__":
    print("🧪 EMG WebSocket Data Stream Verification")
    print("=" * 50)
    
    # 主要測試
    main_uri = "ws://localhost:31278/ws"
    tester = EMGWebSocketTester(main_uri)
    
    success = tester.test_connection(duration=15)  # 測試15秒
    tester.print_summary()
    
    # 如果主要端口沒有數據，測試其他端口
    if tester.data_count == 0:
        working_uri = test_alternative_ports()
        if working_uri:
            print(f"\n💡 建議更新GUI設定為: {working_uri}")
    
    print("\n🏁 Test completed!")