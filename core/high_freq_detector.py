# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - 高頻異常檢測模組

功能 (AUD-005):
- 峰值偵測
- 電感嘯叫 (Coil Whine) 判定
- 高頻共振檢測
- 電子雜訊識別

高頻異常判定標準:
| 檢測項目   | 頻率範圍    | 判定標準      | 說明              |
|------------|-------------|---------------|-------------------|
| 電感嘯叫   | 6k-12k Hz   | 突出量 > 10 dB| GPU/VRM 常見問題  |
| 高頻共振   | 4k-8k Hz    | 有明顯峰值    | 風扇/結構共振     |
| 電子雜訊   | 8k-16k Hz   | 依規格判定    | 電源相關          |
| 超高頻異常 | 16k-20k Hz  | 有明顯峰值    | 年輕人較敏感      |
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from scipy.signal import find_peaks as scipy_find_peaks

# 從 config 導入設定
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import config

# 導入相關模組
from core.filters import highpass_filter, bandpass_filter
from core.fft import compute_average_spectrum, get_frequency_range
from core.noise_level import calculate_rms, rms_to_db


def analyze_high_frequency(
    audio: np.ndarray,
    sample_rate: int,
    cutoff: float = None
) -> Dict[str, Any]:
    """分析高頻成分

    主要分析流程:
    1. 套用高通濾波器
    2. FFT 分析濾波後訊號
    3. 偵測峰值
    4. 判定是否異常 (電感嘯叫等)

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        cutoff: 高通濾波器截止頻率 (Hz)

    Returns:
        Dict: 高頻分析結果
            - band_analysis: 各頻帶分析結果
            - high_freq_peaks: 高頻峰值列表
            - coil_whine_detected: 是否偵測到電感嘯叫
            - coil_whine_frequency: 電感嘯叫頻率
            - coil_whine_prominence: 電感嘯叫突出量
            - possible_cause: 可能原因
            - recommendation: 建議
            - overall_status: 整體狀態 ("PASS", "WARNING", "FAIL")
    """
    if cutoff is None:
        cutoff = config.filter.default_highpass_cutoff

    # 1. 套用高通濾波器
    audio_filtered = highpass_filter(audio, sample_rate, cutoff)

    # 2. FFT 分析
    frequencies, magnitudes_db = compute_average_spectrum(
        audio_filtered, sample_rate
    )

    # 限制到分析範圍
    freq_min, freq_max = config.fft.freq_range
    frequencies, magnitudes_db = get_frequency_range(
        frequencies, magnitudes_db, cutoff, freq_max
    )

    # 3. 偵測峰值
    peaks = detect_peaks(
        frequencies, magnitudes_db,
        prominence=config.high_freq_detection.peak_prominence
    )

    # 4. 分析各頻帶能量
    band_analysis = analyze_band_energy(audio, sample_rate)

    # 5. 檢測電感嘯叫
    coil_whine_result = detect_coil_whine(
        peaks,
        freq_range=config.high_freq_detection.coil_whine_range,
        min_prominence=config.high_freq_detection.coil_whine_prominence
    )

    # 6. 判定整體狀態
    overall_status = determine_status(band_analysis, peaks, coil_whine_result)

    # 7. 生成建議
    recommendation = generate_recommendation(coil_whine_result, peaks, band_analysis)

    return {
        "band_analysis": band_analysis,
        "high_freq_peaks": peaks,
        "coil_whine_detected": coil_whine_result["detected"],
        "coil_whine_frequency": coil_whine_result.get("frequency"),
        "coil_whine_prominence": coil_whine_result.get("prominence"),
        "possible_cause": coil_whine_result.get("possible_cause"),
        "recommendation": recommendation,
        "overall_status": overall_status,
        "filter_cutoff": cutoff
    }


def detect_peaks(
    frequencies: np.ndarray,
    magnitudes_db: np.ndarray,
    prominence: float = None,
    threshold_db: float = -60
) -> List[Dict[str, float]]:
    """偵測頻譜峰值

    Args:
        frequencies: 頻率陣列 (Hz)
        magnitudes_db: 能量陣列 (dB)
        prominence: 最小突出量 (dB)
        threshold_db: 最小能量門檻 (dB)

    Returns:
        List[Dict]: 峰值列表，每個包含:
            - frequency: 頻率 (Hz)
            - magnitude_db: 幅度 (dB)
            - prominence_db: 突出量 (dB)
    """
    if prominence is None:
        prominence = config.high_freq_detection.peak_prominence

    if len(frequencies) == 0 or len(magnitudes_db) == 0:
        return []

    # 使用 scipy 找峰值
    peak_indices, properties = scipy_find_peaks(
        magnitudes_db,
        height=threshold_db,
        prominence=prominence
    )

    # 組裝結果
    peaks = []
    for i, idx in enumerate(peak_indices):
        peaks.append({
            "frequency": float(frequencies[idx]),
            "magnitude_db": float(magnitudes_db[idx]),
            "prominence_db": float(properties["prominences"][i])
        })

    # 按能量排序
    peaks.sort(key=lambda x: x["magnitude_db"], reverse=True)

    return peaks


def detect_coil_whine(
    peaks: List[Dict[str, float]],
    freq_range: Tuple[int, int] = None,
    min_prominence: float = None
) -> Dict[str, Any]:
    """檢測電感嘯叫 (Coil Whine)

    電感嘯叫通常出現在 6k-12kHz 範圍，具有高突出量的特徵。
    常見於 GPU、VRM 等電源管理元件。

    Args:
        peaks: 峰值列表
        freq_range: 檢測頻率範圍 (Hz)
        min_prominence: 最小突出量門檻 (dB)

    Returns:
        Dict: 檢測結果
            - detected: 是否偵測到
            - frequency: 電感嘯叫頻率 (Hz)
            - prominence: 突出量 (dB)
            - possible_cause: 可能原因
    """
    if freq_range is None:
        freq_range = config.high_freq_detection.coil_whine_range
    if min_prominence is None:
        min_prominence = config.high_freq_detection.coil_whine_prominence

    low_freq, high_freq = freq_range

    # 在指定範圍內尋找符合條件的峰值
    for peak in peaks:
        freq = peak["frequency"]
        prom = peak["prominence_db"]

        if low_freq <= freq <= high_freq and prom >= min_prominence:
            # 判斷可能原因
            if 6000 <= freq <= 8000:
                cause = "GPU 電感嘯叫"
            elif 8000 <= freq <= 10000:
                cause = "VRM 電感嘯叫"
            elif 10000 <= freq <= 12000:
                cause = "電源相關電感嘯叫"
            else:
                cause = "電感嘯叫"

            return {
                "detected": True,
                "frequency": freq,
                "prominence": prom,
                "possible_cause": cause
            }

    return {
        "detected": False,
        "frequency": None,
        "prominence": None,
        "possible_cause": None
    }


def analyze_band_energy(
    audio: np.ndarray,
    sample_rate: int
) -> Dict[str, Dict[str, Any]]:
    """分析各頻帶能量

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)

    Returns:
        Dict: 各頻帶分析結果，每個頻帶包含:
            - range: 頻率範圍字串
            - energy_db: 能量值 (dB)
            - status: 狀態 ("PASS", "WARNING", "FAIL")
    """
    results = {}
    nyquist = sample_rate / 2

    for band_name, band_config in config.frequency_bands.bands.items():
        low, high = band_config.range_hz

        # 確保頻率在有效範圍內
        low = max(20, low)
        high = min(high, nyquist - 1)

        if low >= high:
            results[band_name] = {
                "range": f"{low}-{high}Hz",
                "energy_db": -100.0,
                "status": "PASS"
            }
            continue

        try:
            # 濾波後計算能量
            filtered = bandpass_filter(audio, sample_rate, low, high)
            rms = calculate_rms(filtered)
            energy_db = rms_to_db(rms, 1.0)  # 使用相對 dB

            # 確保能量值有效 (不是 -inf 或 NaN)
            if not np.isfinite(energy_db):
                energy_db = -100.0

            # 判定狀態
            status = _determine_band_status(band_name, energy_db)

            results[band_name] = {
                "range": f"{_format_freq(low)}-{_format_freq(high)}",
                "energy_db": round(energy_db, 1),
                "status": status
            }
        except Exception:
            results[band_name] = {
                "range": f"{low}-{high}Hz",
                "energy_db": -100.0,
                "status": "PASS"
            }

    return results


def _format_freq(freq: float) -> str:
    """格式化頻率為可讀字串"""
    if freq >= 1000:
        return f"{freq/1000:.0f}kHz"
    return f"{freq:.0f}Hz"


def _determine_band_status(band_name: str, energy_db: float) -> str:
    """判定頻帶狀態

    根據頻帶類型和能量值判定狀態。
    """
    # 高頻帶的門檻較嚴格
    if band_name in ["high_freq", "ultra_high_freq"]:
        if energy_db > -30:  # 相對較高的高頻能量
            return "WARNING"
        elif energy_db > -20:
            return "FAIL"
    elif band_name == "mid_high_freq":
        if energy_db > -25:
            return "WARNING"

    return "PASS"


def determine_status(
    band_energy: Dict[str, Dict[str, Any]],
    peaks: List[Dict[str, float]],
    coil_whine_result: Dict[str, Any]
) -> str:
    """判定整體狀態

    Args:
        band_energy: 頻帶能量分析結果
        peaks: 峰值列表
        coil_whine_result: 電感嘯叫檢測結果

    Returns:
        str: "PASS", "WARNING", 或 "FAIL"
    """
    # 電感嘯叫優先判定
    if coil_whine_result["detected"]:
        prom = coil_whine_result.get("prominence", 0)
        if prom >= 15:  # 非常明顯的電感嘯叫
            return "FAIL"
        return "WARNING"

    # 檢查各頻帶狀態
    has_warning = False
    for band_name, band_data in band_energy.items():
        status = band_data.get("status", "PASS")
        if status == "FAIL":
            return "FAIL"
        elif status == "WARNING":
            has_warning = True

    # 檢查高頻峰值數量
    high_freq_peaks = [p for p in peaks if p["frequency"] >= 6000]
    if len(high_freq_peaks) >= 3:
        has_warning = True

    return "WARNING" if has_warning else "PASS"


def generate_recommendation(
    coil_whine_result: Dict[str, Any],
    peaks: List[Dict[str, float]],
    band_energy: Dict[str, Dict[str, Any]]
) -> str:
    """生成建議

    根據分析結果生成使用者建議。
    """
    recommendations = []

    if coil_whine_result["detected"]:
        freq = coil_whine_result.get("frequency", 0)
        cause = coil_whine_result.get("possible_cause", "電感嘯叫")
        recommendations.append(
            f"偵測到 {cause} (約 {freq:.0f} Hz)，建議在負載變化時複測確認"
        )

    # 檢查高頻能量
    high_freq_data = band_energy.get("high_freq", {})
    if high_freq_data.get("status") == "WARNING":
        recommendations.append("高頻帶能量偏高，建議檢查電子元件")

    ultra_high_data = band_energy.get("ultra_high_freq", {})
    if ultra_high_data.get("status") == "WARNING":
        recommendations.append("超高頻能量偏高，可能影響年輕使用者")

    if not recommendations:
        return "高頻分析正常，無明顯異常"

    return "；".join(recommendations)
