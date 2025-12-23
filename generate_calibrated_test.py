# -*- coding: utf-8 -*-
"""
生成精確標定的 Leq 測試音檔

這些檔案的 Leq 數值會精確對應檔名。
"""

import numpy as np
import os

# 使用 wave 模組避免依賴問題
import wave
import struct

def generate_calibrated_test_files():
    sample_rate = 48000
    duration = 3.0  # 3 秒
    frequency = 1000.0  # 1kHz (A-weighting 增益 = 0 dB)
    test_dir = "test_data"
    
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # 參考聲壓
    REFERENCE_PRESSURE = 20e-6
    
    # 目標 Leq 列表
    target_leqs = [60, 65, 70, 73, 75, 80, 85, 90]
    
    for leq in target_leqs:
        # 計算對應的 RMS 值
        rms = REFERENCE_PRESSURE * (10 ** (leq / 20))
        # 計算對應的 Peak 振幅
        peak_amplitude = rms * np.sqrt(2)
        
        # 確保振幅不超過 1.0 (數位滿刻度)
        if peak_amplitude > 1.0:
            print(f"⚠️ {leq} dB 的振幅 ({peak_amplitude:.4f}) 超過數位滿刻度，跳過")
            continue
        
        # 生成正弦波
        n_samples = int(sample_rate * duration)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        audio = peak_amplitude * np.sin(2 * np.pi * frequency * t)
        
        # 儲存為 WAV 檔
        filename = f"calibrated_{leq}dB.wav"
        filepath = os.path.join(test_dir, filename)
        
        with wave.open(filepath, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            for sample in audio:
                packed = struct.pack('<h', int(sample * 32767))
                wav_file.writeframes(packed)
        
        print(f"✅ {filename} | Peak={peak_amplitude:.6f} | RMS={rms:.6f} | 預期 Leq={leq} dB")

if __name__ == "__main__":
    print("=== 生成精確標定的 Leq 測試音檔 ===\n")
    generate_calibrated_test_files()
    print("\n完成！請將 test_data 資料夾內的檔案上傳測試。")
