# -*- coding: utf-8 -*-
"""
FFT 計算精度驗證腳本
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.fft import compute_fft, compute_average_spectrum
from core.noise_level import REFERENCE_PRESSURE

def test_fft_accuracy():
    print("=== FFT 計算精度驗證 ===\n")
    
    # 測試參數
    sr = 48000
    duration = 3.0
    freq = 1000  # 1kHz
    
    # 目標 Leq
    target_leq = 70.0
    
    # 計算對應的振幅
    rms = REFERENCE_PRESSURE * (10 ** (target_leq / 20))
    peak_amplitude = rms * np.sqrt(2)
    
    print(f"目標 Leq: {target_leq} dB")
    print(f"計算 RMS: {rms:.6f}")
    print(f"計算 Peak: {peak_amplitude:.6f}")
    
    # 生成正弦波
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = peak_amplitude * np.sin(2 * np.pi * freq * t)
    
    # 測試 compute_fft
    freqs, mags_db = compute_fft(audio, sr)
    
    # 找到 1kHz 附近的峰值
    idx_1k = np.argmin(np.abs(freqs - freq))
    peak_freq = freqs[idx_1k]
    peak_db = mags_db[idx_1k]
    
    print(f"\n--- compute_fft 結果 ---")
    print(f"峰值頻率: {peak_freq:.1f} Hz")
    print(f"峰值幅度: {peak_db:.2f} dB")
    print(f"與目標差異: {peak_db - target_leq:.2f} dB")
    
    # 測試 compute_average_spectrum
    freqs_avg, mags_avg_db = compute_average_spectrum(audio, sr)
    
    idx_1k_avg = np.argmin(np.abs(freqs_avg - freq))
    peak_freq_avg = freqs_avg[idx_1k_avg]
    peak_db_avg = mags_avg_db[idx_1k_avg]
    
    print(f"\n--- compute_average_spectrum 結果 ---")
    print(f"峰值頻率: {peak_freq_avg:.1f} Hz")
    print(f"峰值幅度: {peak_db_avg:.2f} dB")
    print(f"與目標差異: {peak_db_avg - target_leq:.2f} dB")
    
    # 判定結果
    if abs(peak_db_avg - target_leq) < 0.5:
        print("\n✅ FFT 計算精度良好 (誤差 < 0.5 dB)")
    else:
        print(f"\n❌ FFT 計算仍有誤差: {abs(peak_db_avg - target_leq):.2f} dB")

if __name__ == "__main__":
    test_fft_accuracy()
