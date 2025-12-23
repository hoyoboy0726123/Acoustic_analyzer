
import wave
import struct
import math
import os

def create_pure_test_files():
    sample_rate = 48000
    duration = 3.0  # 3秒
    frequency = 1000.0 # 1kHz
    test_dir = "test_data"
    
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    def write_wav(filename, amplitude):
        path = os.path.join(test_dir, filename)
        n_samples = int(sample_rate * duration)
        
        # 開啟 wav 檔案
        with wave.open(path, 'w') as wav_file:
            # 設定參數: 1聲道, 2 bytes (16-bit), 48000Hz
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            
            for i in range(n_samples):
                # 生成正弦波
                value = amplitude * math.sin(2.0 * math.pi * frequency * i / sample_rate)
                # 轉換為 16-bit 整數 (-32768 到 32767)
                packed_value = struct.pack('<h', int(value * 32767))
                wav_file.writeframes(packed_value)
        
        print(f"✅ 已生成 {filename} (Amplitude={amplitude})")

    # 1. 94dB 等級 (全幅值)
    write_wav('test_94dB.wav', 1.0)
    # 2. 74dB 等級 (0.1 幅值)
    write_wav('test_74dB.wav', 0.1)
    # 3. 54dB 等級 (0.01 幅值)
    write_wav('test_54dB.wav', 0.01)

if __name__ == "__main__":
    create_pure_test_files()
    print("\n請將上述檔案上傳至系統，您現在應該會看到正值的 dBA 數據。")
