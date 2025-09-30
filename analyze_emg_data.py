import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_emg_data(csv_file):
    """分析和繪製 EMG 數據"""
    try:
        # 讀取 CSV 數據
        print(f"讀取數據: {csv_file}")
        df = pd.read_csv(csv_file)
        
        print(f"數據形狀: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        print("\n前5行數據:")
        print(df.head())
        
        # 提取通道數據
        channel_1 = df['channel_1'].values
        
        print(f"\n數據統計:")
        print(f"數據點數: {len(channel_1)}")
        print(f"最小值: {channel_1.min():.6f}")
        print(f"最大值: {channel_1.max():.6f}")
        print(f"平均值: {channel_1.mean():.6f}")
        print(f"標準差: {channel_1.std():.6f}")
        
        # 創建簡單的時間-電壓圖
        plt.figure(figsize=(12, 6))
        
        # 創建時間軸（假設採樣頻率，可以根據實際情況調整）
        # 如果有實際的時間戳記錄，可以使用 df['timestamp']
        time_axis = np.arange(len(channel_1))  # 樣本點作為時間軸
        
        plt.plot(time_axis, channel_1, 'b-', linewidth=0.8)
        plt.title('EMG 電壓 vs 時間', fontsize=16)
        plt.xlabel('時間 (樣本點)', fontsize=12)
        plt.ylabel('電壓 (V)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        
        plt.tight_layout()
        
        # 儲存圖表
        output_file = csv_file.replace('.csv', '_voltage_time.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n圖表已儲存: {output_file}")
        
        # 顯示圖表
        plt.show()
        
        return df, channel_1
        
    except FileNotFoundError:
        print(f"❌ 找不到檔案: {csv_file}")
        return None, None
    except Exception as e:
        print(f"❌ 分析過程中發生錯誤: {e}")
        return None, None

if __name__ == "__main__":
    # 分析測試數據
    csv_file = "test_continuous.csv"
    df, data = analyze_emg_data(csv_file)
    
    if data is not None:
        print("\n✅ 分析完成")
        print("查看生成的圖表來判斷數據特性")
    else:
        print("❌ 分析失敗")