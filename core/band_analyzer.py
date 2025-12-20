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
from core.noise_level import calculate_rms, rms_to_db


def compute_octave_bands(
    audio: np.ndarray,
    sample_rate: int,
    use_a_weighting: bool = True,
    n_fft: int = 16384
) -> Dict[str, Any]:
    """計算 1/3 倍頻程頻譜 (FFT Synthesis 方法)

    使用 STFT Average Magnitude Spectrum 合成 1/3 倍頻程能量，
    確保與前端顯示的 FFT 頻譜圖一致。符合 IEC 61260-1 頻帶劃分。

    Args:
        audio: 音訊數據
        sample_rate: 取樣率
        use_a_weighting: 是否應用 A加權 (IEC 61672-1)
        n_fft: FFT 點數

    Returns:
        Dict: 包含分析結果
            - nominal_freqs: 中心頻率列表 (List[float])
            - band_levels: 各頻帶 dB 值 (List[float])
            - raw_max: 未歸一化的最大 dB 值 (用於診斷)
    """
    # 參數驗證
    try:
        sample_rate = float(sample_rate)
        if sample_rate <= 0: raise ValueError
    except:
        sample_rate = 48000.0

    # 確保音訊為 float64
    audio = np.asarray(audio, dtype=np.float64).flatten()
    n_samples = len(audio)

    # 調整 n_fft (確保不過長)
    hop_length = n_fft // 2
    
    if n_samples < n_fft:
        n_fft = n_samples
        hop_length = n_samples
    
    window = np.hanning(n_fft)
    freqs = rfftfreq(n_fft, 1/sample_rate)
    
    # 計算 STFT Average Magnitude
    # -----------------------------------------------------
    n_frames = 1 + (n_samples - n_fft) // hop_length
    if n_frames < 1: n_frames = 1
    
    mag_sum = np.zeros(len(freqs))
    valid_frames = 0
    
    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        
        if end > n_samples: break
             
        frame = audio[start:end] * window
        # FFT Magnitude: |X|/N * 2 (one-sided)
        spec = np.abs(rfft(frame)) / n_fft
        spec[1:-1] *= 2 
        mag_sum += spec
        valid_frames += 1
        
    if valid_frames > 0:
        avg_mag = mag_sum / valid_frames
    else:
        padded = np.zeros(n_fft)
        padded[:n_samples] = audio * np.hanning(n_samples)
        avg_mag = np.abs(rfft(padded)) / n_fft
        avg_mag[1:-1] *= 2

    # A-weighting (IEC 61672-1)
    # -----------------------------------------------------
    def get_a_weighting_gain(f):
        """計算 A-weighting 幅度增益 (linear gain)"""
        f = np.array(f, dtype=float)
        valid = f > 0
        
        f_val = f[valid]
        f2 = f_val**2
        
        # Standard IEC formula applied to Magnitude (Numerator f^2)
        # Result matches 12dB/octave slope at low freq
        num = 12194**2 * f2
        den = (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2)
        
        mag_gain = num / den
        
        # Apply 2.0 dB normalization (gain at 1kHz = 0dB reference)
        # 10^(2.0/20) = 1.2589
        
        full_gain = np.zeros_like(f)
        full_gain[valid] = mag_gain * 1.258925
        return full_gain

    if use_a_weighting:
        weighting = get_a_weighting_gain(freqs)
        weighted_mag = avg_mag * weighting
    else:
        weighted_mag = avg_mag

    # 1/3 Octave Band Integration
    # -----------------------------------------------------
    nominal_freqs = [
        12.5, 16, 20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250,
        315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000,
        5000, 6300, 8000, 10000, 12500, 16000, 20000
    ]
    
    G = 10 ** (3/10)
    FRACTION = 3
    nyquist = sample_rate / 2
    
    band_levels = []
    processed_freqs = []
    
    for fc in nominal_freqs:
        # Band limits
        factor = G ** (1 / (2 * FRACTION))
        f_low = fc / factor
        f_high = fc * factor
        effective_f_high = min(f_high, nyquist)
        
        val_to_append = -120.0
        
        # Cutoff handling
        if f_low > nyquist:
            band_levels.append(-120.0)
            processed_freqs.append(fc)
            continue
            
        indices = np.where((freqs >= f_low) & (freqs < effective_f_high))[0]
        
        if len(indices) > 0:
            # Energy = sum(mag^2)
            band_energy = np.sum(weighted_mag[indices]**2)
            if band_energy > 1e-20:
                val_to_append = 10 * np.log10(band_energy)
        elif f_low < nyquist:
            # Interpolation
            idx = np.abs(freqs - fc).argmin()
            bin_width = sample_rate / n_fft
            band_width = effective_f_high - f_low
            scale = band_width / bin_width
            band_energy = (weighted_mag[idx]**2) * scale
            if band_energy > 1e-20:
                val_to_append = 10 * np.log10(band_energy)
                
        band_levels.append(val_to_append)
        processed_freqs.append(fc)
        
    raw_max = max(band_levels) if band_levels else -120.0
    
    return {
        "nominal_freqs": processed_freqs,
        "band_levels": band_levels,
        "raw_max": raw_max
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
            energy_db = rms_to_db(rms, 1.0)

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
        total_energy_db = 10 * np.log10(total_energy_linear + 1e-10)
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
        energy_db = rms_to_db(rms, 1.0)

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
