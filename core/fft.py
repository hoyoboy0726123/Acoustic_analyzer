# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - FFT 頻譜分析模組

功能 (AUD-002):
- 使用 FFT 將時域音訊轉換為頻域
- 計算各頻率能量分布

演算法參數:
- n_fft: 4096 (FFT 點數)
- hop_length: 512 (幀移動步長)
- window: 'hann' (窗函數)
- freq_range: (20, 20000) Hz
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from scipy import fft as scipy_fft
from scipy.signal import get_window

# 從 config 導入設定
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import config


# 常數定義
EPSILON = 1e-10  # 避免 log(0) 的極小值


def a_weighting(frequencies: np.ndarray) -> np.ndarray:
    """計算 A-weighting 加權係數
    
    根據 IEC 61672-1 標準計算 A-weighting 曲線，模擬人耳對不同頻率的敏感度。
    
    A-weighting 公式:
    Ra(f) = 12194² × f⁴ / ((f² + 20.6²) × sqrt((f² + 107.7²)(f² + 737.9²)) × (f² + 12194²))
    A(f) = 20 × log10(Ra(f)) + 2.0 dB
    
    Args:
        frequencies: 頻率陣列 (Hz)
        
    Returns:
        np.ndarray: A-weighting 加權值 (dB)
    """
    f = np.asarray(frequencies, dtype=float)
    
    # 避免除以零
    f = np.maximum(f, EPSILON)
    
    # IEC 61672-1 A-weighting 係數
    f2 = f ** 2
    
    # A-weighting 計算
    numerator = 12194.0 ** 2 * f2 ** 2
    denominator = ((f2 + 20.6 ** 2) * 
                   np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * 
                   (f2 + 12194.0 ** 2))
    
    ra = numerator / denominator
    
    # 轉為 dB 並加上校正值
    a_weight_db = 20 * np.log10(ra + EPSILON) + 2.0
    
    return a_weight_db


def apply_a_weighting(
    frequencies: np.ndarray, 
    magnitudes_db: np.ndarray
) -> np.ndarray:
    """將 A-weighting 加權套用到頻譜數據
    
    Args:
        frequencies: 頻率陣列 (Hz)
        magnitudes_db: 頻譜幅度 (dB)
        
    Returns:
        np.ndarray: 加權後的頻譜幅度 (dB(A))
    """
    a_weight = a_weighting(frequencies)
    return magnitudes_db + a_weight


def compute_fft(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = None,
    window: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """計算單次 FFT 頻譜

    對整段音訊訊號進行 FFT 分析，使用 scipy.fft.rfft 優化實數訊號處理。

    Args:
        audio: 音訊資料 (1D numpy 陣列)
        sample_rate: 取樣率 (Hz)
        n_fft: FFT 點數，預設使用 config 設定
        window: 窗函數類型，預設使用 config 設定

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - frequencies: 頻率陣列 (Hz)
            - magnitudes_db: 能量陣列 (dB)

    Example:
        >>> import numpy as np
        >>> sr = 48000
        >>> t = np.linspace(0, 1, sr)
        >>> # 產生 1000 Hz 正弦波
        >>> audio = np.sin(2 * np.pi * 1000 * t)
        >>> freqs, mags = compute_fft(audio, sr)
        >>> # 找到最大能量的頻率
        >>> peak_freq = freqs[np.argmax(mags)]
        >>> print(f"峰值頻率: {peak_freq:.0f} Hz")
    """
    # 使用預設值
    if n_fft is None:
        n_fft = config.fft.n_fft
    if window is None:
        window = config.fft.window

    # 確保音訊為 1D
    audio = np.asarray(audio).flatten()

    # 取音訊長度，若比 n_fft 短則用 zero-padding
    n_samples = len(audio)

    # 套用窗函數
    if n_samples >= n_fft:
        # 取中間段
        start = (n_samples - n_fft) // 2
        audio_windowed = audio[start:start + n_fft]
    else:
        # Zero-padding
        audio_windowed = np.zeros(n_fft)
        audio_windowed[:n_samples] = audio

    # 套用窗函數
    win = get_window(window, len(audio_windowed))
    audio_windowed = audio_windowed * win

    # 使用 scipy.fft.rfft 計算實數訊號的 FFT (只回傳正頻率)
    fft_result = scipy_fft.rfft(audio_windowed, n=n_fft)

    # 計算頻率軸
    frequencies = scipy_fft.rfftfreq(n_fft, d=1.0 / sample_rate)

    # 計算幅度 (取絕對值並正規化)
    magnitudes = np.abs(fft_result) / n_fft

    # 雙邊轉單邊 (DC 和 Nyquist 不需要乘 2)
    magnitudes[1:-1] *= 2

    # 轉換為分貝 (dB)
    magnitudes_db = 20 * np.log10(magnitudes + EPSILON)

    return frequencies, magnitudes_db


def compute_stft(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = None,
    hop_length: int = None,
    window: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """計算短時傅立葉變換 (STFT)

    將音訊分割成多個幀，對每幀進行 FFT，得到時頻表示。

    Args:
        audio: 音訊資料 (1D numpy 陣列)
        sample_rate: 取樣率 (Hz)
        n_fft: FFT 點數
        hop_length: 幀移動步長
        window: 窗函數類型

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - times: 時間軸 (秒)
            - frequencies: 頻率軸 (Hz)
            - spectrogram_db: 頻譜圖 (dB)，shape = (n_freqs, n_frames)
    """
    # 使用預設值
    if n_fft is None:
        n_fft = config.fft.n_fft
    if hop_length is None:
        hop_length = config.fft.hop_length
    if window is None:
        window = config.fft.window

    audio = np.asarray(audio).flatten()
    n_samples = len(audio)

    # 計算幀數
    n_frames = 1 + (n_samples - n_fft) // hop_length
    if n_frames < 1:
        n_frames = 1

    # 頻率軸
    frequencies = scipy_fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    n_freqs = len(frequencies)

    # 時間軸
    times = np.arange(n_frames) * hop_length / sample_rate

    # 初始化頻譜圖
    spectrogram = np.zeros((n_freqs, n_frames))

    # 窗函數
    win = get_window(window, n_fft)

    # 逐幀計算 FFT
    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft

        if end <= n_samples:
            frame = audio[start:end] * win
        else:
            # 補零
            frame = np.zeros(n_fft)
            frame[:n_samples - start] = audio[start:] * win[:n_samples - start]

        # FFT
        fft_result = scipy_fft.rfft(frame, n=n_fft)
        magnitudes = np.abs(fft_result) / n_fft
        magnitudes[1:-1] *= 2

        spectrogram[:, i] = magnitudes

    # 轉換為 dB
    spectrogram_db = 20 * np.log10(spectrogram + EPSILON)

    return times, frequencies, spectrogram_db


def get_frequency_range(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    freq_min: float = None,
    freq_max: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """擷取指定頻率範圍的資料

    Args:
        frequencies: 頻率陣列 (Hz)
        magnitudes: 能量陣列 (dB)
        freq_min: 最小頻率 (Hz)，預設使用 config 設定
        freq_max: 最大頻率 (Hz)，預設使用 config 設定

    Returns:
        Tuple[np.ndarray, np.ndarray]: 指定範圍內的 (頻率, 能量)

    Example:
        >>> freqs, mags = compute_fft(audio, sr)
        >>> # 只看 100-5000 Hz
        >>> freqs_range, mags_range = get_frequency_range(freqs, mags, 100, 5000)
    """
    if freq_min is None:
        freq_min = config.fft.freq_range[0]
    if freq_max is None:
        freq_max = config.fft.freq_range[1]

    # 找出範圍內的索引
    mask = (frequencies >= freq_min) & (frequencies <= freq_max)

    return frequencies[mask], magnitudes[mask]


def compute_average_spectrum(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = None,
    hop_length: int = None,
    window: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """計算平均頻譜

    對整段音訊進行 STFT，並對時間軸取平均，得到平均頻譜。
    這對於分析持續性噪音特別有用。

    Args:
        audio: 音訊資料 (1D numpy 陣列)
        sample_rate: 取樣率 (Hz)
        n_fft: FFT 點數
        hop_length: 幀移動步長
        window: 窗函數類型

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - frequencies: 頻率陣列 (Hz)
            - avg_magnitudes_db: 平均能量陣列 (dB)

    Example:
        >>> freqs, avg_mags = compute_average_spectrum(audio, sr)
        >>> import matplotlib.pyplot as plt
        >>> plt.semilogx(freqs, avg_mags)
        >>> plt.xlabel('Frequency (Hz)')
        >>> plt.ylabel('Magnitude (dB)')
    """
    # 計算 STFT
    times, frequencies, spectrogram_db = compute_stft(
        audio, sample_rate, n_fft, hop_length, window
    )

    # 對時間軸取平均
    # 注意：在 dB 域取平均需要先轉回線性域
    spectrogram_linear = 10 ** (spectrogram_db / 20)
    avg_magnitudes_linear = np.mean(spectrogram_linear, axis=1)
    avg_magnitudes_db = 20 * np.log10(avg_magnitudes_linear + EPSILON)

    return frequencies, avg_magnitudes_db


def find_peaks(
    frequencies: np.ndarray,
    magnitudes_db: np.ndarray,
    threshold_db: float = -60,
    min_distance_hz: float = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """找出頻譜中的峰值

    Args:
        frequencies: 頻率陣列 (Hz)
        magnitudes_db: 能量陣列 (dB)
        threshold_db: 最小能量門檻 (dB)
        min_distance_hz: 峰值之間的最小距離 (Hz)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - peak_indices: 峰值索引
            - peak_frequencies: 峰值頻率 (Hz)
            - peak_magnitudes: 峰值能量 (dB)
    """
    from scipy.signal import find_peaks as scipy_find_peaks

    # 計算最小距離的索引數
    if len(frequencies) > 1:
        freq_resolution = frequencies[1] - frequencies[0]
        min_distance_samples = max(1, int(min_distance_hz / freq_resolution))
    else:
        min_distance_samples = 1

    # 使用 scipy 找峰值
    peak_indices, properties = scipy_find_peaks(
        magnitudes_db,
        height=threshold_db,
        distance=min_distance_samples
    )

    peak_frequencies = frequencies[peak_indices]
    peak_magnitudes = magnitudes_db[peak_indices]

    return peak_indices, peak_frequencies, peak_magnitudes


def get_band_energy(
    frequencies: np.ndarray,
    magnitudes_db: np.ndarray,
    low_freq: float,
    high_freq: float
) -> float:
    """計算指定頻帶的總能量

    Args:
        frequencies: 頻率陣列 (Hz)
        magnitudes_db: 能量陣列 (dB)
        low_freq: 頻帶下限 (Hz)
        high_freq: 頻帶上限 (Hz)

    Returns:
        float: 頻帶總能量 (dB)
    """
    # 擷取頻帶範圍
    freqs_band, mags_band = get_frequency_range(
        frequencies, magnitudes_db, low_freq, high_freq
    )

    if len(mags_band) == 0:
        return -100.0  # 沒有資料時回傳極小值

    # 在 dB 域計算總能量需要轉回線性域後相加
    linear_magnitudes = 10 ** (mags_band / 20)
    total_linear = np.sum(linear_magnitudes ** 2)
    total_db = 10 * np.log10(total_linear + EPSILON)

    return total_db


def analyze_spectrum(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int = None,
    freq_range: Tuple[float, float] = None
) -> Dict[str, Any]:
    """完整頻譜分析

    整合多項頻譜分析功能，回傳完整分析結果。

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        n_fft: FFT 點數
        freq_range: 分析頻率範圍 (Hz)

    Returns:
        Dict: 分析結果
            - frequencies: 頻率陣列
            - magnitudes_db: 能量陣列 (dB)
            - peak_frequencies: 峰值頻率列表
            - peak_magnitudes: 峰值能量列表
            - max_frequency: 最大能量的頻率
            - max_magnitude: 最大能量值
            - band_energies: 各頻帶能量
    """
    if freq_range is None:
        freq_range = config.fft.freq_range

    # 計算平均頻譜
    frequencies, magnitudes_db = compute_average_spectrum(
        audio, sample_rate, n_fft
    )

    # 限制頻率範圍
    frequencies, magnitudes_db = get_frequency_range(
        frequencies, magnitudes_db, freq_range[0], freq_range[1]
    )

    # 找峰值
    peak_indices, peak_frequencies, peak_magnitudes = find_peaks(
        frequencies, magnitudes_db
    )

    # 最大能量
    max_idx = np.argmax(magnitudes_db)
    max_frequency = frequencies[max_idx]
    max_magnitude = magnitudes_db[max_idx]

    # 計算各頻帶能量
    band_energies = {}
    for band_name, band_config in config.frequency_bands.bands.items():
        low, high = band_config.range_hz
        energy = get_band_energy(frequencies, magnitudes_db, low, high)
        band_energies[band_name] = {
            "range_hz": (low, high),
            "energy_db": round(energy, 2)
        }

    return {
        "frequencies": frequencies.tolist(),
        "magnitudes_db": magnitudes_db.tolist(),
        "peak_frequencies": peak_frequencies.tolist(),
        "peak_magnitudes": peak_magnitudes.tolist(),
        "max_frequency": float(max_frequency),
        "max_magnitude": float(max_magnitude),
        "band_energies": band_energies,
        "sample_rate": sample_rate,
        "n_fft": n_fft or config.fft.n_fft
    }
