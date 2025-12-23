# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - Discrete Tone 檢測模組

功能 (AUD-004):
- 依據 ECMA-418-1 標準檢測突出單頻成分
- 支援 Prominence Ratio (ΔLp) 和 Tone-to-Noise Ratio (ΔLt) 雙準則

判定標準 (ECMA-418-1):
| 方法                | ≥ 1 kHz 門檻 | < 1 kHz 門檻公式                    |
|---------------------|--------------|-------------------------------------|
| Prominence Ratio    | > 9 dB       | > 9 + 10×log₁₀(1000/f) dB           |
| Tone-to-Noise Ratio | > 8 dB       | 增加 2.5 dB/octave                  |

適用頻率範圍: 89.1 Hz - 11,220 Hz

Note: ECMA-418-1 取代並更新了 ECMA-74 Annex D 的技術內容
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

# ECMA-418-1 頻率範圍
ECMA418_FREQ_MIN = 89.1    # Hz
ECMA418_FREQ_MAX = 11220   # Hz


# ===== ECMA-74 門檻 (舊版，較寬鬆) =====
def get_ecma74_pr_threshold(frequency: float) -> Tuple[float, str]:
    """取得 ECMA-74 Annex D Prominence Ratio 門檻值 (舊版)
    
    使用固定頻帶門檻:
    - 89.1-282 Hz: > 8 dB
    - 282-893 Hz: > 5 dB  
    - 893-11200 Hz: > 3 dB
    
    Args:
        frequency: 頻率 (Hz)
    
    Returns:
        Tuple[float, str]: (門檻 dB, 頻帶名稱)
    """
    if frequency < ECMA418_FREQ_MIN or frequency > ECMA418_FREQ_MAX:
        return None, None
    
    if frequency < 282:
        return 8.0, "89.1-282Hz"
    elif frequency < 893:
        return 5.0, "282-893Hz"
    else:
        return 3.0, "893-11200Hz"


# ===== ECMA-418-1 門檻 (新版，較嚴格) =====
def get_ecma418_pr_threshold(frequency: float) -> Tuple[float, str]:
    """取得 ECMA-418-1 Prominence Ratio (ΔLp) 門檻值 (新版)
    
    公式 (ECMA-418-1):
    - f ≥ 1000 Hz: > 9 dB
    - f < 1000 Hz: > 9 + 10×log₁₀(1000/f) dB
    
    Args:
        frequency: 頻率 (Hz)
    
    Returns:
        Tuple[float, str]: (門檻 dB, 頻帶名稱)
    """
    if frequency < ECMA418_FREQ_MIN or frequency > ECMA418_FREQ_MAX:
        return None, None
    
    if frequency >= 1000:
        threshold = 9.0
        band = "≥1kHz"
    else:
        threshold = 9.0 + 10 * np.log10(1000 / frequency)
        band = "<1kHz"
    
    return threshold, band


def get_prominence_ratio_threshold(frequency: float, standard: str = 'ECMA-74') -> Tuple[float, str]:
    """取得 Prominence Ratio 門檻值 (支援切換標準)
    
    Args:
        frequency: 頻率 (Hz)
        standard: 使用的標準 ('ECMA-74' 或 'ECMA-418')
    
    Returns:
        Tuple[float, str]: (門檻 dB, 頻帶名稱)
    """
    if standard == 'ECMA-418':
        return get_ecma418_pr_threshold(frequency)
    else:  # ECMA-74 (預設)
        return get_ecma74_pr_threshold(frequency)


def get_tone_to_noise_threshold(frequency: float, standard: str = 'ECMA-74') -> Tuple[float, str]:
    """取得 Tone-to-Noise Ratio (ΔLt) 門檻值
    
    ECMA-74: 使用與 PR 相同的固定頻帶門檻
    ECMA-418-1:
    - f ≥ 1000 Hz: > 8 dB
    - f < 1000 Hz: 8 + 2.5×log₂(1000/f) dB
    
    Args:
        frequency: 頻率 (Hz)
        standard: 使用的標準 ('ECMA-74' 或 'ECMA-418')
    
    Returns:
        Tuple[float, str]: (門檻 dB, 方法名稱)
    """
    if frequency < ECMA418_FREQ_MIN or frequency > ECMA418_FREQ_MAX:
        return None, None
    
    if standard == 'ECMA-418':
        if frequency >= 1000:
            threshold = 8.0
        else:
            octaves_below_1k = np.log2(1000 / frequency)
            threshold = 8.0 + 2.5 * octaves_below_1k
    else:  # ECMA-74
        # ECMA-74 的 TNR 使用相同的固定門檻
        if frequency < 282:
            threshold = 8.0
        elif frequency < 893:
            threshold = 5.0
        else:
            threshold = 3.0
    
    return threshold, "TNR"


def get_ecma418_threshold(frequency: float) -> Tuple[float, str]:
    """取得 ECMA-418-1 門檻值 (兼容舊接口)"""
    return get_ecma418_pr_threshold(frequency)


def get_ecma74_threshold(frequency: float) -> Tuple[float, str]:
    """取得 ECMA-74 門檻值 (兼容舊接口)"""
    return get_ecma74_pr_threshold(frequency)


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

    # 計算突出量 (Prominence Ratio)
    prominence = target_level - avg_reference_db

    return float(prominence)


def calculate_tone_to_noise_ratio(
    frequency: float,
    frequencies: np.ndarray,
    magnitudes_db: np.ndarray
) -> float:
    """計算 Tone-to-Noise Ratio (ΔLt)
    
    依據 ECMA-74 Annex D 計算音調對噪音的比值。
    公式：ΔLt = Lt - Ln
    其中 Lt 是音調能量，Ln 是移除音調後的臨界頻帶噪音能量。
    
    Args:
        frequency: 目標頻率 (Hz)
        frequencies: 頻率陣列
        magnitudes_db: 能量陣列 (dB)
    
    Returns:
        float: Tone-to-Noise Ratio (dB)
    """
    if len(frequencies) == 0 or len(magnitudes_db) == 0:
        return 0.0
    
    # 計算臨界頻寬
    cb = calculate_critical_bandwidth(frequency)
    
    # 找到臨界頻帶內的所有頻率點
    low_freq = frequency - cb / 2
    high_freq = frequency + cb / 2
    band_mask = (frequencies >= low_freq) & (frequencies <= high_freq)
    
    if np.sum(band_mask) == 0:
        return 0.0
    
    band_freqs = frequencies[band_mask]
    band_mags = magnitudes_db[band_mask]
    
    # 找到音調（峰值）的索引
    target_idx = np.argmin(np.abs(band_freqs - frequency))
    tone_level = band_mags[target_idx]
    
    # 計算音調的寬度（假設為頻率解析度的 3 倍）
    freq_resolution = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 1
    tone_width = 3 * freq_resolution
    
    # 移除音調區域，計算剩餘噪音
    noise_mask = np.abs(band_freqs - frequency) > tone_width / 2
    
    if np.sum(noise_mask) == 0:
        return 0.0
    
    noise_mags = band_mags[noise_mask]
    
    # 計算噪音的總能量（線性域）
    noise_linear = 10 ** (noise_mags / 10)  # 功率域
    total_noise_power = np.sum(noise_linear)
    noise_level = 10 * np.log10(total_noise_power + 1e-10)
    
    # Tone-to-Noise Ratio
    tnr = tone_level - noise_level
    
    return float(tnr)


def detect_discrete_tones(
    audio: np.ndarray,
    sample_rate: int,
    use_ecma74: bool = True,
    spectrum_mode: str = 'average',
    window_function: str = 'hann',
    n_fft: int = 8192,
    ecma_standard: str = 'ECMA-74',
    use_a_weighting: bool = False
) -> Dict[str, Any]:
    """檢測 Discrete Tones

    支援 ECMA-74 (舊版) 和 ECMA-418 (新版) 標準。

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        use_ecma74: 是否使用 ECMA 標準頻率範圍 (89.1-11200 Hz)
        spectrum_mode: 頻譜分析模式 (average/peak_hold/psd)
        window_function: 窗函數 (hann/hamming/blackman/flattop)
        n_fft: FFT 點數
        ecma_standard: 使用的判定標準 ('ECMA-74' 或 'ECMA-418')
        use_a_weighting: 是否在回傳的頻譜數據中應用 A-weighting (僅用於視覺化，不影響檢測邏輯)

    Returns:
        Dict: 檢測結果
            - tone_detected: bool - 是否偵測到 Tone
            - tones: List[Dict] - 每個 tone 包含相關資訊
            - criteria: str - 使用的判定標準
    """
    # 根據選擇的模式計算頻譜
    if spectrum_mode == 'peak_hold':
        from core.fft import compute_peak_hold_spectrum
        frequencies, magnitudes_db = compute_peak_hold_spectrum(audio, sample_rate, window=window_function, n_fft=n_fft)
    elif spectrum_mode == 'psd':
        from core.fft import compute_psd
        frequencies, magnitudes_db = compute_psd(audio, sample_rate, window=window_function, n_fft=n_fft)
    else:  # average (default)
        frequencies, magnitudes_db = compute_average_spectrum(audio, sample_rate, window=window_function, n_fft=n_fft)

    # 保留完整頻譜用於回傳顯示 (不裁切)
    full_frequencies = frequencies.copy()
    full_magnitudes = magnitudes_db.copy()

    # 限制到分析範圍 (用於檢測邏輯，依據 ECMA 標準)
    if use_ecma74:
        # ECMA 標準範圍
        ecma_freq_min, ecma_freq_max = 89.1, 11200
        frequencies, magnitudes_db = get_frequency_range(
            frequencies, magnitudes_db, ecma_freq_min, ecma_freq_max
        )
    else:
        # 使用設定檔範圍
        freq_min, freq_max = config.fft.freq_range
        frequencies, magnitudes_db = get_frequency_range(
            frequencies, magnitudes_db, freq_min, freq_max
        )

    # 應用 A-weighting 到回傳數據 (如果需要)
    if use_a_weighting:
        from core.fft import a_weighting
        a_weights = a_weighting(full_frequencies)
        full_magnitudes += a_weights

    if len(frequencies) == 0:
        return {"tone_detected": False, "tones": [], "criteria": ecma_standard, "frequencies": [], "magnitudes": []}

    # 找出頻譜中的峰值
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

        # 根據選擇的標準取得門檻
        pr_threshold, band = get_prominence_ratio_threshold(freq, ecma_standard)
        tnr_threshold, _ = get_tone_to_noise_threshold(freq, ecma_standard)

        if pr_threshold is None:
            continue  # 頻率不在範圍內

        # 計算 Prominence Ratio (ΔLp)
        prominence = calculate_tone_prominence(
            freq, frequencies, magnitudes_db
        )
        
        # 計算 Tone-to-Noise Ratio (ΔLt)
        tnr = calculate_tone_to_noise_ratio(
            freq, frequencies, magnitudes_db
        )

        # 雙準則判定：任一超過門檻即為 Prominent
        pr_exceeds = prominence > pr_threshold
        tnr_exceeds = tnr > tnr_threshold
        exceeds = pr_exceeds or tnr_exceeds
        
        # 決定主要判定方法
        if pr_exceeds and tnr_exceeds:
            method = "PR+TNR"
        elif pr_exceeds:
            method = "PR"
        elif tnr_exceeds:
            method = "TNR"
        else:
            method = "-"

        detected_tones.append({
            "frequency": float(freq),
            "prominence": round(float(prominence), 2),
            "tnr": round(float(tnr), 2),
            "magnitude_db": round(float(magnitude), 2),
            "band": band,
            "pr_threshold": round(float(pr_threshold), 2),
            "tnr_threshold": round(float(tnr_threshold), 2),
            "exceeds_threshold": exceeds,
            "method": method
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
        "frequencies": full_frequencies,  # 使用完整頻譜
        "magnitudes": full_magnitudes,    # 使用完整頻譜 (可能已 A-weighted)
        "analysis_range": f"{89.1 if use_ecma74 else config.fft.freq_range[0]:.1f}-{11200 if use_ecma74 else config.fft.freq_range[1]:.0f} Hz",
        "criteria": f"{ecma_standard} (PR+TNR dual criteria)"
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

    # 取得 ECMA-418-1 門檻
    threshold, band = get_ecma418_threshold(actual_freq)

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
