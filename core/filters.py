# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - 數位濾波器模組

功能 (AUD-005):
- 設計與套用 Butterworth 濾波器
- 支援高通、低通、帶通濾波
- 零相位濾波 (filtfilt)

濾波器規格:
- filter_type: 'butterworth'
- filter_order: 5
- implementation: scipy.signal.butter + scipy.signal.filtfilt
- default_highpass_cutoff: 4000 Hz
"""

from typing import Tuple, Literal, Union, List
import numpy as np
from scipy.signal import butter, filtfilt, lfilter, freqz

# 從 config 導入設定
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import config


def design_butterworth_filter(
    cutoff: Union[float, Tuple[float, float]],
    sample_rate: int,
    filter_type: Literal["low", "high", "band"] = "high",
    order: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """設計 Butterworth 濾波器

    Butterworth 濾波器具有最平坦的通帶響應，適合音頻處理。

    Args:
        cutoff: 截止頻率 (Hz)
            - 對於 "low" 和 "high"：單一頻率值
            - 對於 "band"：(低截止頻率, 高截止頻率)
        sample_rate: 取樣率 (Hz)
        filter_type: 濾波器類型
            - "low": 低通濾波器 (保留低於截止頻率的成分)
            - "high": 高通濾波器 (保留高於截止頻率的成分)
            - "band": 帶通濾波器 (保留指定範圍內的成分)
        order: 濾波器階數，預設使用 config 設定

    Returns:
        Tuple[np.ndarray, np.ndarray]: (分子係數 b, 分母係數 a)

    Example:
        >>> b, a = design_butterworth_filter(4000, 48000, "high")
        >>> # 套用濾波器
        >>> y_filtered = filtfilt(b, a, audio)
    """
    if order is None:
        order = config.filter.filter_order

    # 計算正規化截止頻率 (相對於 Nyquist 頻率)
    nyquist = sample_rate / 2

    if filter_type == "band":
        if not isinstance(cutoff, (tuple, list)) or len(cutoff) != 2:
            raise ValueError("帶通濾波器需要 (低截止, 高截止) 頻率對")
        normalized_cutoff = [c / nyquist for c in cutoff]
        btype = 'band'
    else:
        if isinstance(cutoff, (tuple, list)):
            cutoff = cutoff[0]
        normalized_cutoff = cutoff / nyquist
        btype = 'highpass' if filter_type == "high" else 'lowpass'

    # 確保截止頻率在有效範圍內
    if isinstance(normalized_cutoff, list):
        for nc in normalized_cutoff:
            if nc <= 0 or nc >= 1:
                raise ValueError(
                    f"截止頻率必須在 0 到 {nyquist} Hz 之間"
                )
    else:
        if normalized_cutoff <= 0 or normalized_cutoff >= 1:
            raise ValueError(
                f"截止頻率必須在 0 到 {nyquist} Hz 之間"
            )

    # 設計濾波器
    b, a = butter(order, normalized_cutoff, btype=btype)

    return b, a


def apply_filter(
    audio: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    zero_phase: bool = True
) -> np.ndarray:
    """套用濾波器

    Args:
        audio: 音訊資料
        b: 濾波器分子係數
        a: 濾波器分母係數
        zero_phase: 是否使用零相位濾波 (filtfilt)
            - True: 使用 filtfilt，無相位失真但需要更多計算
            - False: 使用 lfilter，有相位延遲但即時處理

    Returns:
        np.ndarray: 濾波後的音訊資料

    Note:
        零相位濾波 (filtfilt) 會前後兩次套用濾波器，
        消除相位延遲，但有效階數會加倍。
    """
    if zero_phase:
        # 零相位濾波：前後各濾波一次
        return filtfilt(b, a, audio)
    else:
        # 一般濾波：會有相位延遲
        return lfilter(b, a, audio)


def highpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    cutoff: float = None,
    order: int = None,
    zero_phase: bool = True
) -> np.ndarray:
    """高通濾波器

    將低於截止頻率的成分濾除，保留高頻成分。
    適用於隔離高頻噪音（如電感嘯叫）進行分析。

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        cutoff: 截止頻率 (Hz)，預設使用 config 設定
        order: 濾波器階數
        zero_phase: 是否使用零相位濾波

    Returns:
        np.ndarray: 濾波後的音訊資料

    Example:
        >>> # 濾除 4kHz 以下的成分
        >>> high_freq = highpass_filter(audio, 48000, cutoff=4000)
    """
    if cutoff is None:
        cutoff = config.filter.default_highpass_cutoff

    b, a = design_butterworth_filter(cutoff, sample_rate, "high", order)
    return apply_filter(audio, b, a, zero_phase)


def lowpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    cutoff: float = 500,
    order: int = None,
    zero_phase: bool = True
) -> np.ndarray:
    """低通濾波器

    將高於截止頻率的成分濾除，保留低頻成分。
    適用於分析低頻噪音（如風扇噪音）。

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        cutoff: 截止頻率 (Hz)
        order: 濾波器階數
        zero_phase: 是否使用零相位濾波

    Returns:
        np.ndarray: 濾波後的音訊資料

    Example:
        >>> # 只保留 500Hz 以下的低頻成分
        >>> low_freq = lowpass_filter(audio, 48000, cutoff=500)
    """
    b, a = design_butterworth_filter(cutoff, sample_rate, "low", order)
    return apply_filter(audio, b, a, zero_phase)


def bandpass_filter(
    audio: np.ndarray,
    sample_rate: int,
    low_cutoff: float,
    high_cutoff: float,
    order: int = None,
    zero_phase: bool = True
) -> np.ndarray:
    """帶通濾波器

    僅保留指定頻率範圍內的成分。
    適用於分析特定頻帶的噪音。

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        low_cutoff: 低截止頻率 (Hz)
        high_cutoff: 高截止頻率 (Hz)
        order: 濾波器階數
        zero_phase: 是否使用零相位濾波

    Returns:
        np.ndarray: 濾波後的音訊資料

    Example:
        >>> # 只保留 1kHz - 4kHz 的成分
        >>> mid_freq = bandpass_filter(audio, 48000, 1000, 4000)
    """
    b, a = design_butterworth_filter(
        (low_cutoff, high_cutoff), sample_rate, "band", order
    )
    return apply_filter(audio, b, a, zero_phase)


def separate_frequency_bands(
    audio: np.ndarray,
    sample_rate: int,
    order: int = None
) -> dict:
    """分離各頻帶

    根據 config 中定義的頻帶，將音訊分離成多個頻帶分量。

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        order: 濾波器階數

    Returns:
        dict: 各頻帶的濾波結果
            - low_freq: 低頻成分
            - mid_freq: 中頻成分
            - mid_high_freq: 中高頻成分
            - high_freq: 高頻成分
            - ultra_high_freq: 超高頻成分
    """
    results = {}
    nyquist = sample_rate / 2

    for band_name, band_config in config.frequency_bands.bands.items():
        low, high = band_config.range_hz

        # 確保頻率在有效範圍內
        low = max(20, low)  # 最低 20 Hz
        high = min(high, nyquist - 1)  # 不超過 Nyquist

        if low >= high:
            # 頻率範圍無效，跳過
            results[band_name] = np.zeros_like(audio)
            continue

        try:
            filtered = bandpass_filter(audio, sample_rate, low, high, order)
            results[band_name] = filtered
        except Exception:
            # 濾波失敗時回傳零
            results[band_name] = np.zeros_like(audio)

    return results


def get_filter_response(
    cutoff: Union[float, Tuple[float, float]],
    sample_rate: int,
    filter_type: Literal["low", "high", "band"] = "high",
    order: int = None,
    n_points: int = 1024
) -> Tuple[np.ndarray, np.ndarray]:
    """取得濾波器頻率響應

    用於視覺化濾波器的頻率響應曲線。

    Args:
        cutoff: 截止頻率 (Hz)
        sample_rate: 取樣率 (Hz)
        filter_type: 濾波器類型
        order: 濾波器階數
        n_points: 頻率點數

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - frequencies: 頻率陣列 (Hz)
            - response_db: 響應曲線 (dB)
    """
    b, a = design_butterworth_filter(cutoff, sample_rate, filter_type, order)

    # 計算頻率響應
    w, h = freqz(b, a, worN=n_points, fs=sample_rate)

    # 轉換為 dB
    epsilon = 1e-10
    response_db = 20 * np.log10(np.abs(h) + epsilon)

    return w, response_db
