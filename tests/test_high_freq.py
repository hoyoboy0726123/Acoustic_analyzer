# -*- coding: utf-8 -*-
"""
高頻檢測模組單元測試 (AUD-005)
驗證 Coil Whine 檢測邏輯
"""

import numpy as np
import sys
# 確保 python 能找到 core 模組
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.high_freq_detector import analyze_high_frequency

class TestHighFreqDetector:
    """高頻檢測測試類別"""

    def test_detect_coil_whine_positive(self):
        """測試明顯的 Coil Whine (電感嘯叫)"""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # 背景噪音 (White Noise, Low Amplitude)
        # RMS = 0.001
        noise = np.random.normal(0, 0.001, len(t))
        
        # Coil Whine: 8500 Hz, Amplitude 0.05
        freq_target = 8500
        coil_whine = 0.05 * np.sin(2 * np.pi * freq_target * t)
        
        audio = noise + coil_whine
        
        # 執行分析
        result = analyze_high_frequency(audio, sr)
        
        # 驗證
        if not result["coil_whine_detected"]:
            raise AssertionError("Should detect coil whine")
            
        print(f"Detected Cause: {result['possible_cause']}")
        
        freq_error = abs(result["coil_whine_frequency"] - freq_target)
        if freq_error >= 50:
            raise AssertionError(f"Freq Error {freq_error}Hz too large")
        
    def test_no_coil_whine(self):
        """測試無嘯叫情況 (純噪音)"""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        
        audio = np.random.normal(0, 0.01, len(t))
        
        result = analyze_high_frequency(audio, sr)
        
        if result["coil_whine_detected"]:
            # check prominence
            print(f"False Positive? Prominence: {result.get('coil_whine_prominence')}")
            raise AssertionError("Should NOT detect coil whine")

    def test_ultra_high_freq_whine(self):
        """測試超高頻異常 (16kHz)"""
        sr = 48000
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        
        noise = np.random.normal(0, 0.001, len(t))
        # 16500 Hz
        whine = 0.05 * np.sin(2 * np.pi * 16500 * t)
        
        audio = noise + whine
        
        result = analyze_high_frequency(audio, sr, cutoff=4000)
        
        peaks = result["high_freq_peaks"]
        target_peak = next((p for p in peaks if abs(p["frequency"] - 16500) < 50), None)
        
        if target_peak is None:
            raise AssertionError("Should detect 16.5kHz peak")
            
        if target_peak["prominence_db"] <= 10:
             raise AssertionError(f"Peak prominence too low: {target_peak['prominence_db']}")

if __name__ == "__main__":
    t = TestHighFreqDetector()
    try:
        t.test_detect_coil_whine_positive()
        print("test_detect_coil_whine_positive: PASS")
    except Exception as e:
        print(f"test_detect_coil_whine_positive: FAIL - {e}")
        import traceback
        traceback.print_exc()

    try:
        t.test_no_coil_whine()
        print("test_no_coil_whine: PASS")
    except Exception as e:
        print(f"test_no_coil_whine: FAIL - {e}")
        import traceback
        traceback.print_exc()
        
    try:
        t.test_ultra_high_freq_whine()
        print("test_ultra_high_freq_whine: PASS")
    except Exception as e:
        print(f"test_ultra_high_freq_whine: FAIL - {e}")
        import traceback
        traceback.print_exc()
