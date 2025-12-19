# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - 核心分析模組套件

此套件包含所有音頻分析的核心演算法實作：
- audio_loader: 音檔載入與驗證
- fft: FFT 頻譜分析
- filters: 數位濾波器
- discrete_tone: Discrete Tone 檢測
- noise_level: 噪音等級計算
- high_freq_detector: 高頻異常檢測
- band_analyzer: 頻帶分離分析
"""

from core.audio_loader import (
    load_audio,
    validate_audio,
    get_audio_metadata,
    load_and_validate,
    load_audio_bytes,
    AudioLoadError,
    AudioValidationError
)

from core.fft import (
    compute_fft,
    compute_stft,
    compute_average_spectrum,
    get_frequency_range,
    find_peaks,
    get_band_energy,
    analyze_spectrum
)

from core.noise_level import (
    calculate_noise_level,
    apply_a_weighting,
    calculate_rms,
    rms_to_db,
    calculate_statistical_levels,
    calculate_leq,
    design_a_weighting_filter,
    get_a_weighting_curve
)

__all__ = [
    # audio_loader
    "load_audio",
    "validate_audio",
    "get_audio_metadata",
    "load_and_validate",
    "load_audio_bytes",
    "AudioLoadError",
    "AudioValidationError",
    # fft
    "compute_fft",
    "compute_stft",
    "compute_average_spectrum",
    "get_frequency_range",
    "find_peaks",
    "get_band_energy",
    "analyze_spectrum",
    # noise_level
    "calculate_noise_level",
    "apply_a_weighting",
    "calculate_rms",
    "rms_to_db",
    "calculate_statistical_levels",
    "calculate_leq",
    "design_a_weighting_filter",
    "get_a_weighting_curve",
]
