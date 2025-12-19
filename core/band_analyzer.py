# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - 頻帶分離分析模組

功能 (AUD-006):
- 將音訊分割成五個頻帶
- 計算各頻帶能量比例
- 提供頻帶特徵分析

頻帶定義:
| 頻帶名稱      | 範圍          | 典型噪音來源        |
|---------------|---------------|---------------------|
| 低頻 (LF)     | 20 - 500 Hz   | 風扇、馬達          |
| 中頻 (MF)     | 500 - 2k Hz   | 人聲、機械運轉      |
| 中高頻 (MHF)  | 2k - 6k Hz    | 鍵盤、聽感敏感區    |
| 高頻 (HF)     | 6k - 12k Hz   | 電感嘯叫            |
| 超高頻 (UHF)  | 12k - 20k Hz  | 高頻電子噪音        |
"""

from typing import Dict, List, Any, Tuple
import numpy as np

# 從 config 導入設定
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import config

# 導入相關模組
from core.filters import bandpass_filter
from core.noise_level import calculate_rms, rms_to_db


def analyze_frequency_bands(
    audio: np.ndarray,
    sample_rate: int
) -> Dict[str, Any]:
    """分析各頻帶能量分布

    將音訊分割成五個頻帶，計算各頻帶的能量和佔比。

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)

    Returns:
        Dict: 頻帶分析結果
            - bands: 各頻帶詳細資訊
            - total_energy_db: 總能量 (dB)
            - dominant_band: 主導頻帶
            - energy_distribution: 能量分布百分比
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
    """計算單一頻帶能量

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        low_freq: 頻帶下限 (Hz)
        high_freq: 頻帶上限 (Hz)

    Returns:
        Dict: 頻帶能量
            - energy_db: 能量 (dB)
            - rms: RMS 值
    """
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
    """取得頻帶特徵描述

    根據能量分布生成特徵描述。

    Args:
        band_analysis: analyze_frequency_bands 的輸出

    Returns:
        Dict: 特徵描述
            - profile: 噪音輪廓描述
            - concerns: 潛在問題
            - recommendations: 建議
    """
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
