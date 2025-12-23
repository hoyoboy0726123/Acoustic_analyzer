# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - 頻帶分離分析模組

功能 (AUD-006):
- 1/3 倍頻程全頻段分析 (IEC 61260-1)
- 簡易 5 頻帶能量分布分析 (Summary)
- 提供頻帶特徵分析

頻帶定義 (簡易版):
| 頻帶名稱      | 範圍          | 典型噪音來源        |
|---------------|---------------|---------------------|
| 低頻 (LF)     | 20 - 500 Hz   | 風扇、馬達          |
| 中頻 (MF)     | 500 - 2k Hz   | 人聲、機械運轉      |
| 中高頻 (MHF)  | 2k - 6k Hz    | 鍵盤、聽感敏感區    |
| 高頻 (HF)     | 6k - 12k Hz   | 電感嘯叫            |
| 超高頻 (UHF)  | 12k - 20k Hz  | 高頻電子噪音        |
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from scipy.fft import rfft, rfftfreq

# 從 config 導入設定
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import config

# 導入相關模組
from core.filters import bandpass_filter
from core.noise_level import calculate_rms, rms_to_db, REFERENCE_PRESSURE, EPSILON


def compute_octave_bands(
    audio: np.ndarray,
    sample_rate: int,
    use_a_weighting: bool = True,
    filter_order: int = 6
) -> Dict[str, Any]:
    """計算 1/3 倍頻程頻譜 (IEC 61260-1:2014 濾波器法)

    使用 Butterworth 帶通濾波器組實現，符合 IEC 61260-1:2014 和 ANSI S1.11 標準。
    與 HEAD acoustics ArtemiS SUITE 的 "1/n Octave (Filter)" 方法一致。

    Args:
        audio: 音訊數據
        sample_rate: 取樣率
        use_a_weighting: 是否應用 A加權 (IEC 61672-1)
        filter_order: 濾波器階數 (IEC 61260 建議 6 階)

    Returns:
        Dict: 包含分析結果
            - nominal_freqs: 中心頻率列表 (List[float])
            - band_levels: 各頻帶 dB(SPL) 值 (List[float])
            - method: 使用的分析方法
    """
    from scipy.signal import butter, sosfilt
    
    # 參數驗證
    try:
        sample_rate = float(sample_rate)
        if sample_rate <= 0: raise ValueError
    except:
        sample_rate = 48000.0

    # 確保音訊為 float64
    audio = np.asarray(audio, dtype=np.float64).flatten()
    nyquist = sample_rate / 2

    # IEC 61260-1:2014 標準 1/3 倍頻程中心頻率
    # 完整的標稱頻率序列
    nominal_freqs = [
        12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
        315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
        5000, 6300, 8000, 10000, 12500, 16000, 20000
    ]

    # IEC 61260 頻帶計算常數
    G = 10 ** (3/10)  # Octave ratio = 10^(3/10) ≈ 1.9953
    FRACTION = 3  # 1/3 octave

    # A-weighting 查找表 (IEC 61672-1 at nominal frequencies)
    a_weighting_table = {
        12.5: -63.4, 16: -56.7, 20: -50.5, 25: -44.7, 31.5: -39.4,
        40: -34.6, 50: -30.2, 63: -26.2, 80: -22.5, 100: -19.1,
        125: -16.1, 160: -13.4, 200: -10.9, 250: -8.6, 315: -6.6,
        400: -4.8, 500: -3.2, 630: -1.9, 800: -0.8, 1000: 0.0,
        1250: 0.6, 1600: 1.0, 2000: 1.2, 2500: 1.3, 3150: 1.2,
        4000: 1.0, 5000: 0.5, 6300: -0.1, 8000: -1.1, 10000: -2.5,
        12500: -4.3, 16000: -6.6, 20000: -9.3
    }

    band_levels = []
    processed_freqs = []

    for fc in nominal_freqs:
        # 根據 IEC 61260-1 計算頻帶邊界
        factor = G ** (1 / (2 * FRACTION))  # 約 1.122
        f_low = fc / factor
        f_high = fc * factor

        # 檢查是否在有效範圍內
        if f_low >= nyquist:
            band_levels.append(-120.0)
            processed_freqs.append(fc)
            continue

        # 限制上限頻率在 Nyquist 以下
        effective_f_high = min(f_high, nyquist * 0.95)
        
        # 確保帶寬有效
        if effective_f_high <= f_low:
            band_levels.append(-120.0)
            processed_freqs.append(fc)
            continue

        try:
            # 設計 Butterworth 帶通濾波器 (IEC 61260 建議 6 階)
            # 使用 SOS (Second-Order Sections) 格式以提高穩定性
            sos = butter(
                filter_order,
                [f_low / nyquist, effective_f_high / nyquist],
                btype='bandpass',
                output='sos'
            )

            # 濾波
            filtered_audio = sosfilt(sos, audio)

            # 計算 RMS 能量
            rms = np.sqrt(np.mean(filtered_audio ** 2))
            
            # 轉換為 dB SPL
            if rms > 1e-20:
                level_db = 20 * np.log10(rms / REFERENCE_PRESSURE + EPSILON)
            else:
                level_db = -120.0

            # 套用 A-weighting (如果需要)
            if use_a_weighting:
                a_weight = a_weighting_table.get(fc, 0.0)
                level_db += a_weight

            band_levels.append(round(level_db, 1))

        except Exception as e:
            # 濾波器設計失敗 (通常是極端頻率)
            band_levels.append(-120.0)

        processed_freqs.append(fc)

    return {
        "nominal_freqs": processed_freqs,
        "band_levels": band_levels,
        "method": "IEC 61260-1:2014 Filter Bank (6th order Butterworth)"
    }


def analyze_frequency_bands(
    audio: np.ndarray,
    sample_rate: int
) -> Dict[str, Any]:
    """分析各頻帶能量分布 (5頻帶簡易版)

    將音訊分割成五個寬頻帶 (Low, Mid, ...)，計算各頻帶能量和佔比。
    """
    nyquist = sample_rate / 2
    bands_result = {}
    energies_linear = []
    band_names = []

    # 分析各頻帶
    for band_name, band_config in config.frequency_bands.bands.items():
        low, high = band_config.range_hz

        # 確保頻率在有效範圍內
        low = max(20, low)
        high = min(high, nyquist - 1)

        if low >= high:
            bands_result[band_name] = {
                "range_hz": (low, high),
                "energy_db": -100.0,
                "energy_linear": 0.0,
                "percentage": 0.0,
                "description": band_config.description
            }
            energies_linear.append(0.0)
            band_names.append(band_name)
            continue

        try:
            # 濾波並計算能量
            filtered = bandpass_filter(audio, sample_rate, low, high)
            rms = calculate_rms(filtered)
            energy_db = rms_to_db(rms, REFERENCE_PRESSURE)

            # 確保有效值
            if not np.isfinite(energy_db):
                energy_db = -100.0
                energy_linear = 0.0
            else:
                energy_linear = rms ** 2  # 能量 = RMS^2

            bands_result[band_name] = {
                "range_hz": (low, high),
                "energy_db": round(energy_db, 1),
                "energy_linear": energy_linear,
                "description": band_config.description
            }
            energies_linear.append(energy_linear)
            band_names.append(band_name)

        except Exception:
            bands_result[band_name] = {
                "range_hz": (low, high),
                "energy_db": -100.0,
                "energy_linear": 0.0,
                "description": band_config.description
            }
            energies_linear.append(0.0)
            band_names.append(band_name)

    # 計算總能量和百分比
    total_energy_linear = sum(energies_linear)

    if total_energy_linear > 0:
        for i, band_name in enumerate(band_names):
            percentage = (energies_linear[i] / total_energy_linear) * 100
            bands_result[band_name]["percentage"] = round(percentage, 1)

        # 總能量 dB
        total_energy_db = 10 * np.log10(total_energy_linear / (REFERENCE_PRESSURE**2) + EPSILON)
    else:
        for band_name in band_names:
            bands_result[band_name]["percentage"] = 0.0
        total_energy_db = -100.0

    # 找出主導頻帶
    max_energy_idx = np.argmax(energies_linear)
    dominant_band = band_names[max_energy_idx] if energies_linear[max_energy_idx] > 0 else None

    # 生成能量分布摘要
    energy_distribution = {
        band_name: bands_result[band_name]["percentage"]
        for band_name in band_names
    }

    return {
        "bands": bands_result,
        "total_energy_db": round(total_energy_db, 1),
        "dominant_band": dominant_band,
        "energy_distribution": energy_distribution
    }


def calculate_band_energy(
    audio: np.ndarray,
    sample_rate: int,
    low_freq: float,
    high_freq: float
) -> Dict[str, float]:
    """計算單一頻帶能量"""
    try:
        filtered = bandpass_filter(audio, sample_rate, low_freq, high_freq)
        rms = calculate_rms(filtered)
        energy_db = rms_to_db(rms, REFERENCE_PRESSURE)

        if not np.isfinite(energy_db):
            energy_db = -100.0

        return {
            "energy_db": round(energy_db, 1),
            "rms": float(rms)
        }
    except Exception:
        return {
            "energy_db": -100.0,
            "rms": 0.0
        }


def get_band_characteristics(band_analysis: Dict[str, Any]) -> Dict[str, str]:
    """取得頻帶特徵描述"""
    dist = band_analysis.get("energy_distribution", {})
    dominant = band_analysis.get("dominant_band")

    # 分析噪音輪廓
    lf_pct = dist.get("low_freq", 0)
    mf_pct = dist.get("mid_freq", 0)
    hf_pct = dist.get("high_freq", 0) + dist.get("ultra_high_freq", 0)

    # 判斷噪音類型
    if lf_pct > 50:
        profile = "低頻主導型噪音 (風扇/馬達為主)"
    elif hf_pct > 30:
        profile = "高頻主導型噪音 (可能有電子元件噪音)"
    elif mf_pct > 40:
        profile = "中頻主導型噪音 (機械運轉聲)"
    else:
        profile = "均衡型噪音譜"

    # 潛在問題
    concerns = []
    if hf_pct > 25:
        concerns.append("高頻能量佔比較高，可能存在電感嘯叫")
    if lf_pct > 60:
        concerns.append("低頻能量佔比過高，可能風扇運轉較大")
    if dist.get("ultra_high_freq", 0) > 10:
        concerns.append("超高頻能量偏高，年輕使用者可能感知明顯")

    # 建議
    recommendations = []
    if dominant == "high_freq" or dominant == "ultra_high_freq":
        recommendations.append("建議進行高頻音隔離分析確認來源")
    if dominant == "low_freq":
        recommendations.append("建議檢查散熱系統運轉狀態")

    return {
        "profile": profile,
        "concerns": concerns if concerns else ["無明顯異常"],
        "recommendations": recommendations if recommendations else ["維持目前狀態"]
    }


def apply_band_rejection(
    audio: np.ndarray,
    sample_rate: int,
    removed_bands: List[float],
    fraction: int = 3,
    filter_order: int = 4
) -> np.ndarray:
    """應用多個帶阻濾波器移除指定頻帶"""
    from scipy.signal import butter, sosfiltfilt
    
    if not removed_bands:
        return audio
        
    # 複製避免修改原陣列
    filtered_audio = audio.copy()
    nyquist = sample_rate / 2
    G = 10 ** (3/10)
    
    for fc in removed_bands:
        try:
            # 計算頻帶範圍 (1/3 Octave)
            factor = G ** (1 / (2 * fraction))
            f_low = fc / factor
            f_high = fc * factor
            
            # 正規化頻率檢查
            if f_low >= nyquist:
                continue
                
            if f_high >= nyquist * 0.99:
                # 情況 1: 上限超過 Nyquist -> 轉為低通濾波器 (保留 < f_low)
                # 移除 [f_low, inf]
                wn = min(0.99, f_low / nyquist)
                sos = butter(filter_order, wn, btype='lowpass', output='sos')
            elif f_low <= 0:
                # 情況 2: 下限低於 0 -> 轉為高通濾波器 (保留 > f_high)
                # 移除 [0, f_high]
                wn = min(0.99, max(0.001, f_high / nyquist))
                sos = butter(filter_order, wn, btype='highpass', output='sos')
            else:
                # 情況 3: 正常帶阻濾波器
                wn_low = max(0.001, f_low / nyquist)
                wn_high = min(0.99, f_high / nyquist)
                
                if wn_low >= wn_high: continue
                
                sos = butter(
                    filter_order, 
                    [wn_low, wn_high], 
                    btype='bandstop', 
                    output='sos'
                )
            
            # 濾波 (零相位)
            filtered_audio = sosfiltfilt(sos, filtered_audio)
        except Exception as e:
            # print(f"Filter error at {fc}Hz: {e}") # Debug
            continue

            
    return filtered_audio
