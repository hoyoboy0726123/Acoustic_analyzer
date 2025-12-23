
import numpy as np
import sys
from pathlib import Path

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent))

# 直接從 core 導入計算函數
from core.noise_level import calculate_noise_level, REFERENCE_PRESSURE

def test_precision():
    print("=== 聲學計算邏輯驗證 (修正後版本) ===")
    
    sr = 48000
    duration = 1.0
    freq = 1000  
    amplitude = 0.1  # 數位峰值為 0.1
    
    # 手動生成正弦波
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = amplitude * np.sin(2 * np.pi * freq * t)
    
    # 執行分析 (不使用 A-weighting)
    # 此函數現在應使用 20uPa 作為參考
    result = calculate_noise_level(audio, sr, apply_weighting=False)
    
    calculated_leq = result['leq_dba']
    
    # 理論計算:
    # RMS = 0.1 / sqrt(2) = 0.07071
    # SPL = 20 * log10(0.07071 / 0.00002) = 70.97
    expected_rms = amplitude / np.sqrt(2)
    expected_db = 20 * np.log10(expected_rms / REFERENCE_PRESSURE)
    
    print(f"測試訊號: 1kHz Sine Wave (Amplitude={amplitude})")
    print(f"理論計算 Leq: {expected_db:.2f} dB")
    print(f"系統計算 Leq: {calculated_leq:.2f} dB")
    
    diff = abs(calculated_leq - expected_db)
    if diff < 0.1:
        print("\n✅ [成功] 數學公式已修正。現在 0.1 RMS 對應的是約 71 dB 的正值音壓。")
    else:
        print(f"\n❌ [失敗] 誤差值 {diff:.2f} dB 仍然存在問題。")

if __name__ == "__main__":
    test_precision()
