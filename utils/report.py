# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - 報告生成模組 (AUD-008)

功能:
- 彙整各個分析模組的結果
- 生成 Excel 格式的測試報告 (含詳細 Raw Data)
"""

import pandas as pd
import io
import os
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import numpy as np

# Import core analysis functions
from core.noise_level import calculate_noise_level
from core.band_analyzer import compute_octave_bands
from core.high_freq_detector import analyze_high_frequency
from core.fft import compute_average_spectrum

def generate_excel_report(
    audio_data: np.ndarray,
    sample_rate: int,
    filename: str = "audio.wav"
) -> Tuple[Optional[bytes], Optional[str]]:
    """生成 Excel 測試報告 (Enhanced)
    
    Args:
        audio_data: 音訊數據
        sample_rate: 取樣率
        filename: 原始檔名
        
    Returns:
        Tuple[bytes, str]: (Excel檔案二進位數據, 錯誤訊息/None)
    """
    try:
        # 1. 計算全域噪音指標 (AUD-003) + Time Profile
        # Note: calculate_noise_level now returns 'profile' key
        noise_metrics = calculate_noise_level(audio_data, sample_rate)
        
        # 2. 計算 1/3 倍頻程數據 (AUD-006)
        octave_data = compute_octave_bands(audio_data, sample_rate, use_a_weighting=True)
        
        # 3. 高頻分析與 Coil Whine (AUD-005)
        high_freq_result = analyze_high_frequency(audio_data, sample_rate)
        
        # 4. 計算原始 FFT 頻譜 (Raw Data)
        # 使用 Average 模式
        fft_freqs, fft_mags = compute_average_spectrum(audio_data, sample_rate)

        # --- 建構 DataFrame ---
        
        # Sheet 1: Test Summary
        duration = len(audio_data) / sample_rate
        
        cw_detected = high_freq_result['coil_whine_detected']
        cw_freq = high_freq_result.get('coil_whine_frequency')
        cw_prom = high_freq_result.get('coil_whine_prominence')
        
        status = "PASS"
        if cw_detected:
            status = "WARNING (Coil Whine)"
        elif high_freq_result.get("overall_status") == "WARNING":
            status = "WARNING (High Freq)"
            
        summary_data = {
            "Parameter": [
                "Filename", 
                "Analysis Date", 
                "Overall Status",
                "Sample Rate", 
                "Duration",
                "Leq (A-weighted)", 
                "Lmax", 
                "Lmin", 
                "L10 (90th percentile)",
                "L90 (Background)",
                "Coil Whine Detected", 
                "CW Frequency", 
                "CW Prominence", 
                "Possible Cause"
            ],
            "Value": [
                filename, 
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                status,
                f"{sample_rate} Hz",
                f"{duration:.2f} sec",
                f"{noise_metrics['leq_dba']:.1f} dB",
                f"{noise_metrics['lmax_dba']:.1f} dB",
                f"{noise_metrics['lmin_dba']:.1f} dB",
                f"{noise_metrics['l10']:.1f} dB",
                f"{noise_metrics['l90']:.1f} dB",
                "YES" if cw_detected else "NO",
                f"{cw_freq:.0f} Hz" if cw_detected and cw_freq else "N/A",
                f"{cw_prom:.1f} dB" if cw_detected and cw_prom else "N/A",
                high_freq_result.get('possible_cause', "N/A")
            ],
            "Description": [
                "原始檔案名稱",
                "報告生成時間",
                "自動判定結果",
                "每秒取樣點數",
                "檔案總長度",
                "等效連續音壓級 (A加權)",
                "最大瞬間音壓",
                "最小背景音壓",
                "突發噪音指標 (超過10%時間)",
                "背景噪音指標 (超過90%時間)",
                "是否檢測到電感嘯叫",
                "主要異音頻率",
                "異音突出周圍程度",
                "可能的原因分析"
            ]
        }
        df_summary = pd.DataFrame(summary_data)
        
        # Sheet 2: 1/3 Octave Bands
        df_octave = pd.DataFrame({
            "Center Frequency (Hz)": octave_data["nominal_freqs"],
            "Level dB(A)": [round(x, 2) for x in octave_data["band_levels"]]
        })
        
        # Sheet 3: High Freq Peaks
        peaks = high_freq_result.get("high_freq_peaks", [])
        if peaks:
            df_peaks = pd.DataFrame(peaks)
            df_peaks = df_peaks.rename(columns={
                "frequency": "Frequency (Hz)",
                "magnitude_db": "Magnitude (dB)",
                "prominence_db": "Prominence (dB)"
            })
            df_peaks = df_peaks[["Frequency (Hz)", "Magnitude (dB)", "Prominence (dB)"]].round(2)
        else:
            df_peaks = pd.DataFrame(columns=["Frequency (Hz)", "Magnitude (dB)", "Prominence (dB)"])

        # Sheet 4: Level vs Time (Profile)
        profile = noise_metrics.get("profile", {})
        if profile:
            df_profile = pd.DataFrame({
                "Time (sec)": [round(t, 3) for t in profile.get("times", [])],
                "Level dB(A)": [round(l, 2) for l in profile.get("levels", [])]
            })
        else:
            df_profile = pd.DataFrame(columns=["Time (sec)", "Level dB(A)"])

        # Sheet 5: Raw FFT Spectrum
        df_spectrum = pd.DataFrame({
            "Frequency (Hz)": [round(f, 2) for f in fft_freqs],
            "Magnitude (dB)": [round(m, 2) for m in fft_mags]
        })
        # Filter negative infinity if any
        df_spectrum = df_spectrum[df_spectrum["Magnitude (dB)"] > -200]

        # --- Write to Excel ---
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='Test Summary', index=False)
            df_octave.to_excel(writer, sheet_name='1_3 Octave Bands', index=False)
            df_profile.to_excel(writer, sheet_name='Level vs Time', index=False)
            df_peaks.to_excel(writer, sheet_name='High Freq Peaks', index=False)
            df_spectrum.to_excel(writer, sheet_name='Raw FFT Spectrum', index=False)
            
            # Format Columns
            for sheet_name in writer.sheets:
                sheet = writer.sheets[sheet_name]
                for column in sheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50) # Cap width
                    sheet.column_dimensions[column_letter].width = adjusted_width
            
        return output.getvalue(), None

    except Exception as e:
        import traceback
        return None, f"報告生成失敗: {str(e)}"
