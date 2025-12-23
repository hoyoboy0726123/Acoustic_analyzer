# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - 配置檔

此檔案包含系統所有可配置參數，包括：
- FFT 分析參數
- 濾波器設定
- 頻帶定義
- 檔案驗證規則
- 高頻異常判定標準
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class FFTConfig:
    """FFT 頻譜分析配置

    Attributes:
        n_fft: FFT 點數，決定頻率解析度
        hop_length: 幀移動步長
        window: 窗函數類型
        freq_range: 分析頻率範圍 (Hz)
    """
    n_fft: int = 4096
    hop_length: int = 512
    window: str = 'hann'
    freq_range: Tuple[int, int] = (20, 20000)


@dataclass(frozen=True)
class FilterConfig:
    """數位濾波器配置

    Attributes:
        filter_type: 濾波器類型
        filter_order: 濾波器階數
        default_highpass_cutoff: 預設高通濾波器截止頻率 (Hz)
    """
    filter_type: str = 'butterworth'
    filter_order: int = 5
    default_highpass_cutoff: int = 4000


@dataclass(frozen=True)
class FrequencyBand:
    """單一頻帶定義

    Attributes:
        name: 頻帶名稱
        range_hz: 頻率範圍 (Hz)
        sources: 常見噪音來源
        importance: 重要性等級
    """
    name: str
    range_hz: Tuple[int, int]
    sources: str
    importance: str


@dataclass
class FrequencyBandsConfig:
    """頻帶分離配置

    定義各頻帶的頻率範圍、常見噪音來源及重要性
    """
    bands: Dict[str, FrequencyBand] = field(default_factory=lambda: {
        'low_freq': FrequencyBand(
            name='低頻',
            range_hz=(20, 500),
            sources='風扇轉動、硬碟馬達',
            importance='背景'
        ),
        'mid_freq': FrequencyBand(
            name='中頻',
            range_hz=(500, 2000),
            sources='風切聲、軸承噪音',
            importance='背景'
        ),
        'mid_high_freq': FrequencyBand(
            name='中高頻',
            range_hz=(2000, 6000),
            sources='風扇共振、結構共振',
            importance='注意'
        ),
        'high_freq': FrequencyBand(
            name='高頻',
            range_hz=(6000, 12000),
            sources='電感嘯叫 (Coil Whine)',
            importance='重要'
        ),
        'ultra_high_freq': FrequencyBand(
            name='超高頻',
            range_hz=(12000, 20000),
            sources='電子雜訊、電源相關',
            importance='注意'
        )
    })


@dataclass(frozen=True)
class AudioFileConfig:
    """音檔驗證配置

    Attributes:
        allowed_formats: 允許的檔案格式
        sample_rates: 支援的取樣率 (Hz)
        bit_depths: 支援的位元深度
        max_channels: 最大聲道數
        max_file_size_mb: 最大檔案大小 (MB)
        duration_range: 允許的音檔長度範圍 (秒)
    """
    allowed_formats: Tuple[str, ...] = ('wav', 'mp3', 'flac')
    sample_rates: Tuple[int, ...] = (44100, 48000)
    bit_depths: Tuple[int, ...] = (16, 24)
    max_channels: int = 1  # 僅支援 Mono
    max_file_size_mb: int = 50
    duration_range: Tuple[float, float] = (10.0, 120.0)


@dataclass(frozen=True)
class DiscreteToneThreshold:
    """Discrete Tone 判定門檻 (ECMA-418-1)

    Attributes:
        freq_range: 頻率範圍 (Hz)
        prominence_threshold: 突出量門檻 (dB)
    """
    freq_range: Tuple[float, float]
    prominence_threshold: float


@dataclass
class DiscreteToneConfig:
    """Discrete Tone 檢測配置 (依據 ECMA-418-1)"""
    # ECMA-418-1 使用 PR/TNR 方法，門檻由頻率公式動態計算
    # 此處保留頻帶劃分供參考用途
    thresholds: List[DiscreteToneThreshold] = field(default_factory=lambda: [
        DiscreteToneThreshold(freq_range=(89.1, 282), prominence_threshold=8.0),
        DiscreteToneThreshold(freq_range=(282, 893), prominence_threshold=5.0),
        DiscreteToneThreshold(freq_range=(893, 11200), prominence_threshold=3.0)
    ])


@dataclass(frozen=True)
class HighFreqDetectionConfig:
    """高頻異常檢測配置

    Attributes:
        coil_whine_range: 電感嘯叫頻率範圍 (Hz)
        coil_whine_prominence: 電感嘯叫突出量門檻 (dB)
        resonance_range: 高頻共振頻率範圍 (Hz)
        electronic_noise_range: 電子雜訊頻率範圍 (Hz)
        ultra_high_range: 超高頻異常頻率範圍 (Hz)
        peak_prominence: 一般峰值偵測突出量門檻 (dB)
    """
    coil_whine_range: Tuple[int, int] = (6000, 12000)
    coil_whine_prominence: float = 10.0
    resonance_range: Tuple[int, int] = (4000, 8000)
    electronic_noise_range: Tuple[int, int] = (8000, 16000)
    ultra_high_range: Tuple[int, int] = (16000, 20000)
    peak_prominence: float = 6.0


@dataclass(frozen=True)
class VisualizationConfig:
    """視覺化配置

    Attributes:
        spectrum_figsize: 頻譜圖尺寸 (英吋)
        spectrogram_figsize: 瀑布圖尺寸 (英吋)
        cmap: 顏色映射
        y_axis: Y 軸類型
        fmax: 最大顯示頻率 (Hz)
    """
    spectrum_figsize: Tuple[int, int] = (12, 6)
    spectrogram_figsize: Tuple[int, int] = (12, 6)
    cmap: str = 'viridis'
    y_axis: str = 'log'
    fmax: int = 20000


@dataclass
class AppConfig:
    """應用程式主配置類別

    整合所有子配置，提供統一的配置存取介面
    """
    fft: FFTConfig = field(default_factory=FFTConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    frequency_bands: FrequencyBandsConfig = field(default_factory=FrequencyBandsConfig)
    audio_file: AudioFileConfig = field(default_factory=AudioFileConfig)
    discrete_tone: DiscreteToneConfig = field(default_factory=DiscreteToneConfig)
    high_freq_detection: HighFreqDetectionConfig = field(default_factory=HighFreqDetectionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # API 設定
    api_prefix: str = "/api/v1"
    debug: bool = False

    # 音訊參考值
    reference_pressure: float = 20e-6  # 參考聲壓 20 μPa


# 全域配置實例
config = AppConfig()
