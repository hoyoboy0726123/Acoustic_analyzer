# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - Discrete Tone 檢測模組

功能 (AUD-004):
- 依據 ECMA-74 Annex D 標準檢測突出單頻成分

判定標準 (ECMA-74):
| 頻率範圍       | 突出量門檻 | 判定            |
|----------------|------------|-----------------|
| 89.1 - 282 Hz  | > 8 dB     | Tone Detected   |
| 282 - 893 Hz   | > 5 dB     | Tone Detected   |
| 893 - 11200 Hz | > 3 dB     | Tone Detected   |
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from scipy.signal import find_peaks as scipy_find_peaks

# 從 config 導入設定
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import config

# 導入相關模組
from core.fft import compute_average_spectrum, get_frequency_range


# ECMA-74 Annex D 門檻值定義
ECMA74_THRESHOLDS = [
    {"freq_range": (89.1, 282), "threshold_db": 8.0, "band": "89.1-282Hz"},
    {"freq_range": (282, 893), "threshold_db": 5.0, "band": "282-893Hz"},
    {"freq_range": (893, 11200), "threshold_db": 3.0, "band": "893-11200Hz"},
]


def get_ecma74_threshold(frequency: float) -> Tuple[float, str]:
    """取得 ECMA-74 門檻值

    根據頻率範圍回傳對應的突出量門檻。

    Args:
        frequency: 頻率 (Hz)

    Returns:
        Tuple[float, str]: (突出量門檻 dB, 頻帶名稱)
            若頻率不在任何頻帶內，回傳 (None, None)
    """
    for threshold_def in ECMA74_THRESHOLDS:
        low, high = threshold_def["freq_range"]
        if low <= frequency <= high:
            return threshold_def["threshold_db"], threshold_def["band"]

    return None, None


def calculate_critical_bandwidth(frequency: float) -> float:
    """計算臨界頻寬 (Critical Bandwidth)

    臨界頻寬是人耳聽覺系統的頻率解析度特性。
    使用 Zwicker 公式計算。

    Args:
        frequency: 中心頻率 (Hz)

    Returns:
        float: 臨界頻寬 (Hz)
    """
    # Zwicker 公式的簡化版本
    # CB = 25 + 75 * (1 + 1.4 * (f/1000)^2)^0.69
    f_khz = frequency / 1000
    cb = 25 + 75 * (1 + 1.4 * (f_khz ** 2)) ** 0.69
    return cb


def calculate_tone_prominence(
    frequency: float,
    frequencies: np.ndarray,
    magnitudes_db: np.ndarray,
    use_critical_bandwidth: bool = True
) -> float:
    """計算 Tone 突出量

    計算指定頻率相對於周圍頻帶的突出程度。
    使用臨界頻寬內的平均能量作為參考。

    Args:
        frequency: 目標頻率 (Hz)
        frequencies: 頻率陣列
        magnitudes_db: 能量陣列 (dB)
        use_critical_bandwidth: 是否使用臨界頻寬計算

    Returns:
        float: 突出量 (dB)
    """
    if len(frequencies) == 0 or len(magnitudes_db) == 0:
        return 0.0

    # 找到最接近目標頻率的索引
    target_idx = np.argmin(np.abs(frequencies - frequency))
    target_level = magnitudes_db[target_idx]

    # 計算參考頻帶
    if use_critical_bandwidth:
        cb = calculate_critical_bandwidth(frequency)
    else:
        # 使用固定的 1/3 八度頻寬
        cb = frequency * (2 ** (1/6) - 2 ** (-1/6))

    # 計算頻帶範圍
    low_freq = frequency - cb
    high_freq = frequency + cb

    # 排除目標頻率附近的區域 (避免自己影響自己)
    exclusion_range = cb * 0.1  # 排除臨界頻寬的 10%

    # 計算頻帶內的平均能量 (排除目標頻率)
    mask_low = (frequencies >= low_freq) & (frequencies < frequency - exclusion_range)
    mask_high = (frequencies > frequency + exclusion_range) & (frequencies <= high_freq)
    mask = mask_low | mask_high

    if np.sum(mask) == 0:
        return 0.0

    # 計算平均能量 (需要在線性域計算)
    reference_levels = magnitudes_db[mask]
    reference_linear = 10 ** (reference_levels / 20)
    avg_reference_linear = np.mean(reference_linear)
    avg_reference_db = 20 * np.log10(avg_reference_linear + 1e-10)

    # 計算突出量
    prominence = target_level - avg_reference_db

    return float(prominence)


def detect_discrete_tones(
    audio: np.ndarray,
    sample_rate: int,
    use_ecma74: bool = True
) -> Dict[str, Any]:
    """檢測 Discrete Tones

    依據 ECMA-74 Annex D 標準，在頻譜中尋找突出的單頻成分。

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        use_ecma74: 是否使用 ECMA-74 標準門檻

    Returns:
        Dict: 檢測結果
            - tone_detected: bool - 是否偵測到 Tone
            - tones: List[Dict] - 每個 tone 包含:
                - frequency: 頻率 (Hz)
                - prominence: 突出量 (dB)
                - band: 所屬頻帶
                - threshold: 該頻帶的門檻值
                - exceeds_threshold: 是否超過門檻

    Example:
        >>> result = detect_discrete_tones(audio, 48000)
        >>> if result["tone_detected"]:
        ...     for tone in result["tones"]:
        ...         print(f"Tone at {tone['frequency']:.0f} Hz, prominence: {tone['prominence']:.1f} dB")
    """
    # 計算平均頻譜
    frequencies, magnitudes_db = compute_average_spectrum(audio, sample_rate)

    # 限制到分析範圍 (89.1 - 11200 Hz for ECMA-74)
    if use_ecma74:
        freq_min, freq_max = 89.1, 11200
    else:
        freq_min, freq_max = config.fft.freq_range

    frequencies, magnitudes_db = get_frequency_range(
        frequencies, magnitudes_db, freq_min, freq_max
    )

    if len(frequencies) == 0:
        return {"tone_detected": False, "tones": []}

    # 找出頻譜中的峰值
    # 使用較低的突出量作為初步篩選
    peak_indices, properties = scipy_find_peaks(
        magnitudes_db,
        prominence=2.0,  # 初步篩選：突出量至少 2 dB
        distance=5  # 峰值之間至少間隔 5 個頻率點
    )

    # 分析每個峰值
    detected_tones = []

    for i, idx in enumerate(peak_indices):
        freq = frequencies[idx]
        magnitude = magnitudes_db[idx]

        # 取得該頻率的 ECMA-74 門檻
        threshold, band = get_ecma74_threshold(freq)

        if threshold is None:
            continue  # 頻率不在 ECMA-74 範圍內

        # 計算精確的突出量
        prominence = calculate_tone_prominence(
            freq, frequencies, magnitudes_db
        )

        # 判定是否超過門檻
        exceeds = prominence > threshold

        detected_tones.append({
            "frequency": float(freq),
            "prominence": round(float(prominence), 2),
            "magnitude_db": round(float(magnitude), 2),
            "band": band,
            "threshold": threshold,
            "exceeds_threshold": exceeds
        })

    # 過濾出超過門檻的 Tone (排序按頻率)
    exceeding_tones = [t for t in detected_tones if t["exceeds_threshold"]]
    exceeding_tones.sort(key=lambda x: x["frequency"])

    # 所有偵測到的 tone (不論是否超過門檻，按突出量排序)
    detected_tones.sort(key=lambda x: x["prominence"], reverse=True)

    return {
        "tone_detected": len(exceeding_tones) > 0,
        "tones": exceeding_tones,
        "all_candidates": detected_tones[:10],  # 最多回傳前 10 個候選
        "analysis_range": f"{freq_min:.1f}-{freq_max:.0f} Hz"
    }


def analyze_tone_characteristics(
    audio: np.ndarray,
    sample_rate: int,
    target_frequency: float
) -> Dict[str, Any]:
    """分析特定頻率的 Tone 特性

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        target_frequency: 目標頻率 (Hz)

    Returns:
        Dict: Tone 特性分析結果
    """
    # 計算平均頻譜
    frequencies, magnitudes_db = compute_average_spectrum(audio, sample_rate)

    # 找到最接近目標頻率的索引
    target_idx = np.argmin(np.abs(frequencies - target_frequency))
    actual_freq = frequencies[target_idx]
    magnitude = magnitudes_db[target_idx]

    # 計算突出量
    prominence = calculate_tone_prominence(
        actual_freq, frequencies, magnitudes_db
    )

    # 取得 ECMA-74 門檻
    threshold, band = get_ecma74_threshold(actual_freq)

    # 計算臨界頻寬
    critical_bw = calculate_critical_bandwidth(actual_freq)

    return {
        "target_frequency": target_frequency,
        "actual_frequency": float(actual_freq),
        "magnitude_db": round(float(magnitude), 2),
        "prominence_db": round(float(prominence), 2),
        "critical_bandwidth": round(critical_bw, 2),
        "ecma74_band": band,
        "ecma74_threshold": threshold,
        "exceeds_ecma74": prominence > threshold if threshold else None
    }
