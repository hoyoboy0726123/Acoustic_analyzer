# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - 噪音等級計算模組

功能 (AUD-003):
- 計算 A 加權聲壓級，符合 ECMA-74 標準

計算流程:
1. 對音訊套用 A-weighting 濾波器
2. 計算 RMS 值
3. 轉換為分貝: dB = 20 × log₁₀(rms / ref), ref = 20 μPa
4. 輸出 Leq、Lmax、Lmin、L10、L90

輸出:
- leq_dba: 等效連續音壓級
- lmax_dba: 最大音壓級
- lmin_dba: 最小音壓級
- l10: L10 統計音壓級 (超過 10% 時間的音壓級)
- l90: L90 統計音壓級 (超過 90% 時間的音壓級，代表背景噪音)
"""

from typing import Dict, Tuple, Optional
import numpy as np
from scipy.signal import bilinear, lfilter

# 從 config 導入設定
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import config


# 參考聲壓 (Pa) - 20 μPa
REFERENCE_PRESSURE = 20e-6

# 避免 log(0) 的極小值
EPSILON = 1e-10


def design_a_weighting_filter(sample_rate: int) -> Tuple[np.ndarray, np.ndarray]:
    """設計 A-weighting 數位濾波器

    根據 IEC 61672-1:2013 標準設計 A-weighting 濾波器。
    使用雙線性變換將類比濾波器轉換為數位濾波器。

    Args:
        sample_rate: 取樣率 (Hz)

    Returns:
        Tuple[np.ndarray, np.ndarray]: (分子係數 b, 分母係數 a)

    Note:
        A-weighting 濾波器模擬人耳對不同頻率的敏感度，
        在低頻和高頻處衰減，在 1-4 kHz 處增益最大。
    """
    # A-weighting 濾波器的極點頻率 (Hz)
    # 根據 IEC 61672-1 標準
    f1 = 20.598997
    f2 = 107.65265
    f3 = 737.86223
    f4 = 12194.217

    # 增益常數 (1 kHz 處增益為 0 dB)
    A1000 = 1.9997

    # 計算正規化角頻率
    pi = np.pi

    # 類比濾波器的分子和分母係數
    # A-weighting 濾波器傳遞函數的設計
    # 使用二階區段 (Second-Order Sections) 實現

    # 類比原型濾波器
    # H(s) = K * s^4 / ((s+ω1)^2 * (s+ω2) * (s+ω3) * (s+ω4)^2)

    # 將類比濾波器轉換為數位濾波器
    # 使用雙線性變換

    # 預畸變頻率
    Wc1 = 2 * pi * f1
    Wc2 = 2 * pi * f2
    Wc3 = 2 * pi * f3
    Wc4 = 2 * pi * f4

    # 類比濾波器的分子 (四個微分器)
    num_analog = np.array([A1000 * Wc4**2, 0, 0, 0, 0])

    # 類比濾波器的分母
    # (s + Wc1)^2 * (s + Wc2) * (s + Wc3) * (s + Wc4)^2
    den1 = np.convolve([1, Wc1], [1, Wc1])  # (s + Wc1)^2
    den2 = [1, Wc2]  # (s + Wc2)
    den3 = [1, Wc3]  # (s + Wc3)
    den4 = np.convolve([1, Wc4], [1, Wc4])  # (s + Wc4)^2

    den_analog = np.convolve(den1, den2)
    den_analog = np.convolve(den_analog, den3)
    den_analog = np.convolve(den_analog, den4)

    # 使用雙線性變換將類比濾波器轉換為數位濾波器
    b, a = bilinear(num_analog, den_analog, sample_rate)

    return b, a


def apply_a_weighting(
    audio: np.ndarray,
    sample_rate: int
) -> np.ndarray:
    """套用 A-weighting 濾波器

    A-weighting 是模擬人耳對不同頻率敏感度的濾波曲線。
    人耳對 1-4 kHz 最敏感，對低頻和高頻較不敏感。

    Args:
        audio: 音訊資料 (1D numpy 陣列)
        sample_rate: 取樣率 (Hz)

    Returns:
        np.ndarray: A-weighted 音訊資料

    Example:
        >>> import numpy as np
        >>> sr = 48000
        >>> audio = np.random.randn(sr)  # 1 秒白噪音
        >>> audio_weighted = apply_a_weighting(audio, sr)
    """
    # 設計 A-weighting 濾波器
    b, a = design_a_weighting_filter(sample_rate)

    # 套用濾波器
    audio_weighted = lfilter(b, a, audio)

    return audio_weighted


def calculate_rms(audio: np.ndarray) -> float:
    """計算 RMS (Root Mean Square)

    RMS 是音訊訊號能量的衡量指標。

    Args:
        audio: 音訊資料

    Returns:
        float: RMS 值

    Formula:
        RMS = sqrt(mean(audio^2))
    """
    return np.sqrt(np.mean(audio ** 2))


def rms_to_db(
    rms: float,
    reference: float = REFERENCE_PRESSURE
) -> float:
    """將 RMS 轉換為分貝 (dB)

    使用公式: dB = 20 × log₁₀(rms / reference)

    Args:
        rms: RMS 值
        reference: 參考值 (預設 20 μPa)

    Returns:
        float: 分貝值 (dB)

    Note:
        對於數位音訊，通常使用相對於滿量程的 dBFS。
        若要轉換為實際 SPL，需要麥克風校準資訊。
    """
    if rms <= 0:
        return -np.inf

    return 20 * np.log10(rms / reference + EPSILON)


def calculate_frame_levels(
    audio: np.ndarray,
    sample_rate: int,
    frame_length: int = 4096,
    apply_weighting: bool = True
) -> np.ndarray:
    """計算每個時間幀的音壓級

    將音訊分割成多個幀，計算每幀的 RMS 並轉換為 dB。

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        frame_length: 幀長度 (樣本數)
        apply_weighting: 是否套用 A-weighting

    Returns:
        np.ndarray: 每幀的音壓級 (dB)
    """
    # 是否套用 A-weighting
    if apply_weighting:
        audio = apply_a_weighting(audio, sample_rate)

    # 計算幀數
    n_samples = len(audio)
    n_frames = n_samples // frame_length

    if n_frames == 0:
        # 音訊太短，直接計算整段
        rms = calculate_rms(audio)
        return np.array([rms_to_db(rms, 1.0)])  # 使用相對 dB

    # 計算每幀的音壓級
    frame_levels = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * frame_length
        end = start + frame_length
        frame = audio[start:end]
        rms = calculate_rms(frame)
        # 使用 1.0 作為參考值計算相對 dB (dBFS)
        frame_levels[i] = rms_to_db(rms, 1.0)

    return frame_levels


def calculate_statistical_levels(
    db_values: np.ndarray
) -> Dict[str, float]:
    """計算統計音壓級

    計算 L10, L50, L90 等統計音壓級。
    Ln 表示超過 n% 時間的音壓級。

    Args:
        db_values: 各時間幀的分貝值陣列

    Returns:
        Dict: 統計音壓級
            - l10: 超過 10% 時間的音壓級 (較高的噪音峰值)
            - l50: 超過 50% 時間的音壓級 (中位數)
            - l90: 超過 90% 時間的音壓級 (背景噪音)

    Note:
        L10 通常代表環境中的突發噪音
        L90 通常代表背景噪音水平
        L50 是中位數，代表典型噪音水平
    """
    # 過濾掉無效值
    valid_values = db_values[np.isfinite(db_values)]

    if len(valid_values) == 0:
        return {
            "l10": -np.inf,
            "l50": -np.inf,
            "l90": -np.inf
        }

    # 計算百分位數
    # L10 = 90th percentile (超過 10% 時間 = 前 90%)
    # L50 = 50th percentile
    # L90 = 10th percentile (超過 90% 時間 = 前 10%)
    l10 = np.percentile(valid_values, 90)
    l50 = np.percentile(valid_values, 50)
    l90 = np.percentile(valid_values, 10)

    return {
        "l10": float(round(l10, 1)),
        "l50": float(round(l50, 1)),
        "l90": float(round(l90, 1))
    }


def calculate_leq(db_values: np.ndarray) -> float:
    """計算等效連續音壓級 (Leq)

    Leq 是將時變噪音轉換為穩態噪音的等效能量值。

    Args:
        db_values: 各時間幀的分貝值陣列

    Returns:
        float: Leq 值 (dB)

    Formula:
        Leq = 10 × log₁₀(mean(10^(Li/10)))
    """
    # 過濾掉無效值
    valid_values = db_values[np.isfinite(db_values)]

    if len(valid_values) == 0:
        return -np.inf

    # 計算能量平均
    # 先將 dB 轉換為線性能量，取平均後再轉回 dB
    linear_values = 10 ** (valid_values / 10)
    mean_energy = np.mean(linear_values)
    leq = 10 * np.log10(mean_energy + EPSILON)

    return float(round(leq, 1))


def calculate_noise_level(
    audio: np.ndarray,
    sample_rate: int,
    frame_length: int = 4096,
    apply_weighting: bool = True
) -> Dict[str, float]:
    """計算噪音等級 dB(A)

    完整的噪音等級分析，包含 Leq、Lmax、Lmin 和統計音壓級。

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        frame_length: 分析幀長度 (樣本數)
        apply_weighting: 是否套用 A-weighting

    Returns:
        Dict: 噪音等級計算結果
            - leq_dba: 等效連續音壓級
            - lmax_dba: 最大音壓級
            - lmin_dba: 最小音壓級
            - l10: L10 統計音壓級
            - l90: L90 統計音壓級

    Example:
        >>> import numpy as np
        >>> sr = 48000
        >>> duration = 10  # 秒
        >>> # 模擬噪音訊號
        >>> audio = np.random.randn(sr * duration) * 0.01
        >>> result = calculate_noise_level(audio, sr)
        >>> print(f"Leq: {result['leq_dba']:.1f} dB(A)")
    """
    # 計算每幀的音壓級
    frame_levels = calculate_frame_levels(
        audio, sample_rate, frame_length, apply_weighting
    )

    # 計算 Leq
    leq = calculate_leq(frame_levels)

    # 計算 Lmax, Lmin
    valid_levels = frame_levels[np.isfinite(frame_levels)]

    if len(valid_levels) > 0:
        lmax = float(round(np.max(valid_levels), 1))
        lmin = float(round(np.min(valid_levels), 1))
    else:
        lmax = -np.inf
        lmin = -np.inf

    # 計算統計音壓級
    stat_levels = calculate_statistical_levels(frame_levels)
    
    # 準備時間軸數據
    dt = frame_length / sample_rate
    times = np.arange(len(frame_levels)) * dt

    return {
        "leq_dba": leq,
        "lmax_dba": lmax,
        "lmin_dba": lmin,
        "l10": stat_levels["l10"],
        "l90": stat_levels["l90"],
        # 原始 Profile 數據 (用於報告與繪圖)
        "profile": {
            "times": times,
            "levels": frame_levels
        }
    }


def get_a_weighting_curve(
    sample_rate: int,
    n_points: int = 1024
) -> Tuple[np.ndarray, np.ndarray]:
    """取得 A-weighting 頻率響應曲線

    用於視覺化 A-weighting 濾波器的頻率響應。

    Args:
        sample_rate: 取樣率 (Hz)
        n_points: 頻率點數

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - frequencies: 頻率陣列 (Hz)
            - response_db: 響應曲線 (dB)
    """
    from scipy.signal import freqz

    b, a = design_a_weighting_filter(sample_rate)

    # 計算頻率響應
    w, h = freqz(b, a, worN=n_points, fs=sample_rate)

    # 轉換為 dB
    response_db = 20 * np.log10(np.abs(h) + EPSILON)

    return w, response_db
