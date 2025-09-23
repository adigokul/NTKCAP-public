import websocket
import json
import time

def read_raw_emg_data():
    """簡單版本：直接讀取 EMG WebSocket 數據並輸出原始數據"""
    uri = "ws://localhost:31278/ws"
    
    print(f"連接到: {uri}")
    print("="*50)
    
    try:
        ws = websocket.WebSocket()
        ws.connect(uri)
        print("✅ WebSocket 連接成功")
        print("正在接收數據...\n")
        
        data_count = 0
        
        while True:
            try:
                # 接收原始數據
                raw_data = ws.recv()
                data_count += 1
                
                print(f"數據包 #{data_count}")
                print(f"原始數據長度: {len(raw_data)} bytes")
                
                # 嘗試解析 JSON
                try:
                    data_dict = json.loads(raw_data)
                    print("JSON 解析成功:")
                    
                    # 檢查數據結構
                    if "contents" in data_dict:
                        contents = data_dict["contents"]
                        print(f"  - contents 數量: {len(contents)}")
                        
                        for i, item in enumerate(contents):
                            print(f"  - 項目 {i}:")
                            if "serial_number" in item:
                                print(f"    * serial_number: {item['serial_number']}")
                            if "eeg" in item:
                                eeg_data = item["eeg"]
                                print(f"    * eeg 通道數: {len(eeg_data)}")
                                print(f"    * eeg 數據: {eeg_data[:5]}..." if len(eeg_data) > 5 else f"    * eeg 數據: {eeg_data}")
                                
                    else:
                        print(f"  - 數據結構: {list(data_dict.keys())}")
                        for key, value in data_dict.items():
                            if isinstance(value, list):
                                print(f"    * {key}: 列表長度 {len(value)}")
                            else:
                                print(f"    * {key}: {value}")
                                
                except json.JSONDecodeError as e:
                    print(f"❌ JSON 解析失敗: {e}")
                    print(f"原始數據前100字符: {raw_data[:100]}")
                    
                print("-" * 30)
                time.sleep(0.1)  # 稍微延遲避免輸出太快
                
                # 限制輸出數量避免過多
                if data_count >= 10:
                    print("\n已顯示10個數據包，繼續接收中...")
                    print("按 Ctrl+C 停止程式")
                    data_count = 0  # 重置計數
                    
            except websocket.WebSocketTimeoutException:
                print("⚠️  WebSocket 超時，重新嘗試...")
                continue
                
    except websocket.WebSocketException as e:
        print(f"❌ WebSocket 錯誤: {e}")
    except ConnectionRefusedError:
        print("❌ 連接被拒絕，請確保 EMG 服務器正在運行")
    except KeyboardInterrupt:
        print("\n\n⚠️  程式被用戶中斷")
    except Exception as e:
        print(f"❌ 未預期錯誤: {e}")
    finally:
        try:
            ws.close()
            print("WebSocket 連接已關閉")
        except:
            pass

def test_connection_only():
    """僅測試連接"""
    uri = "ws://localhost:31278/ws"
    
    try:
        print(f"測試連接: {uri}")
        ws = websocket.WebSocket()
        ws.settimeout(5)
        ws.connect(uri)
        print("✅ 連接測試成功")
        ws.close()
        return True
    except Exception as e:
        print(f"❌ 連接測試失敗: {e}")
        return False

if __name__ == "__main__":
    print("EMG WebSocket 簡單數據讀取器")
    print("="*40)
    
    # 先測試連接
    if test_connection_only():
        print("\n開始讀取數據...")
        read_raw_emg_data()
    else:
        print("\n無法連接到 EMG 服務器")
        print("請確保:")
        print("1. EMG 設備已連接")
        print("2. WebSocket 服務器在 localhost:31278 運行")
        print("3. 防火牆允許連接")
        
    input("\n按 Enter 鍵退出...")