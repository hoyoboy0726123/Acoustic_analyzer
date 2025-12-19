# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - 音檔載入與驗證模組

功能 (AUD-001):
- 接收使用者上傳的音訊檔案
- 驗證格式與品質
- 回傳 metadata

支援格式: WAV (必須), MP3, FLAC (可選)
取樣率: 44100 或 48000 Hz
位元深度: 16-bit 或 24-bit
聲道: Mono
最大檔案大小: 50MB
長度: 10-120 秒
"""

from typing import Tuple, Optional, Dict, Any, Union
from pathlib import Path
import os

import numpy as np
import librosa
import soundfile as sf

# 從 config 導入設定
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import config


class AudioLoadError(Exception):
    """音檔載入錯誤"""
    pass


class AudioValidationError(Exception):
    """音檔驗證錯誤"""
    pass


def load_audio(
    file_path: Union[str, Path],
    target_sr: Optional[int] = None,
    mono: bool = True
) -> Tuple[np.ndarray, int]:
    """載入音檔

    使用 librosa 載入音訊檔案，支援 WAV、MP3、FLAC 格式。

    Args:
        file_path: 音檔路徑
        target_sr: 目標取樣率，None 則保留原始取樣率
        mono: 是否轉換為單聲道

    Returns:
        Tuple[np.ndarray, int]: (音訊資料, 取樣率)
            - 音訊資料為 numpy float32 陣列，範圍 [-1.0, 1.0]
            - 取樣率為 Hz

    Raises:
        FileNotFoundError: 找不到檔案
        AudioLoadError: 無法載入音檔
    """
    file_path = Path(file_path)

    # 驗證檔案存在
    if not file_path.exists():
        raise FileNotFoundError(f"找不到音檔: {file_path}")

    # 驗證檔案格式
    suffix = file_path.suffix.lower().lstrip('.')
    if suffix not in config.audio_file.allowed_formats:
        raise AudioLoadError(
            f"不支援的檔案格式: {suffix}。"
            f"支援格式: {', '.join(config.audio_file.allowed_formats)}"
        )

    try:
        # 使用 librosa 載入音檔
        # sr=None 保留原始取樣率，避免自動重新取樣
        y, sr = librosa.load(
            str(file_path),
            sr=target_sr,  # None = 保留原始取樣率
            mono=mono,
            dtype=np.float32
        )

        return y, sr

    except Exception as e:
        raise AudioLoadError(f"無法載入音檔: {e}") from e


def get_audio_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """取得音檔 metadata

    使用 soundfile 讀取音檔的詳細 metadata。

    Args:
        file_path: 音檔路徑

    Returns:
        Dict: 音檔 metadata
            - sample_rate: int - 取樣率 (Hz)
            - channels: int - 聲道數
            - frames: int - 總幀數
            - duration: float - 長度 (秒)
            - format: str - 檔案格式
            - subtype: str - 子類型 (包含位元深度資訊)
            - bit_depth: int - 位元深度

    Raises:
        FileNotFoundError: 找不到檔案
        AudioLoadError: 無法讀取 metadata
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"找不到音檔: {file_path}")

    try:
        # 使用 soundfile 讀取 metadata
        info = sf.info(str(file_path))

        # 解析位元深度
        bit_depth = _parse_bit_depth(info.subtype)

        return {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration": info.duration,
            "format": info.format,
            "subtype": info.subtype,
            "bit_depth": bit_depth,
            "file_size_bytes": file_path.stat().st_size,
            "file_size_mb": file_path.stat().st_size / (1024 * 1024)
        }

    except Exception as e:
        raise AudioLoadError(f"無法讀取音檔 metadata: {e}") from e


def _parse_bit_depth(subtype: str) -> int:
    """解析 soundfile subtype 取得位元深度

    Args:
        subtype: soundfile 回傳的 subtype 字串

    Returns:
        int: 位元深度 (8, 16, 24, 32)
    """
    subtype_upper = subtype.upper()

    if "PCM_16" in subtype_upper:
        return 16
    elif "PCM_24" in subtype_upper:
        return 24
    elif "PCM_32" in subtype_upper:
        return 32
    elif "PCM_S8" in subtype_upper or "PCM_U8" in subtype_upper:
        return 8
    elif "FLOAT" in subtype_upper:
        return 32
    elif "DOUBLE" in subtype_upper:
        return 64
    else:
        # 對於 MP3、FLAC 等壓縮格式，預設回傳 16
        return 16


def validate_audio(
    file_path: Union[str, Path],
    strict: bool = True
) -> Dict[str, Any]:
    """驗證音檔格式與品質

    檢查音檔是否符合系統要求的規格。

    Args:
        file_path: 音檔路徑
        strict: 是否嚴格驗證 (取樣率、位元深度等)

    Returns:
        Dict: 驗證結果與 metadata
            - file_valid: bool - 是否有效
            - sample_rate: int - 取樣率
            - duration: float - 長度
            - channels: int - 聲道數
            - bit_depth: int - 位元深度
            - file_size_mb: float - 檔案大小 (MB)
            - error_message: Optional[str] - 錯誤訊息
            - warnings: List[str] - 警告訊息

    Example:
        >>> result = validate_audio("test.wav")
        >>> if result["file_valid"]:
        ...     print(f"有效音檔，長度: {result['duration']} 秒")
        ... else:
        ...     print(f"無效: {result['error_message']}")
    """
    file_path = Path(file_path)
    errors = []
    warnings = []

    # 基本結果結構
    result = {
        "file_valid": False,
        "sample_rate": 0,
        "duration": 0.0,
        "channels": 0,
        "bit_depth": 0,
        "file_size_mb": 0.0,
        "error_message": None,
        "warnings": []
    }

    # 1. 檢查檔案是否存在
    if not file_path.exists():
        result["error_message"] = f"找不到檔案: {file_path}"
        return result

    # 2. 檢查檔案格式
    suffix = file_path.suffix.lower().lstrip('.')
    if suffix not in config.audio_file.allowed_formats:
        result["error_message"] = (
            f"不支援的檔案格式: {suffix}。"
            f"支援格式: {', '.join(config.audio_file.allowed_formats)}"
        )
        return result

    # 3. 檢查檔案大小
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    result["file_size_mb"] = round(file_size_mb, 2)

    if file_size_mb > config.audio_file.max_file_size_mb:
        result["error_message"] = (
            f"檔案太大: {file_size_mb:.1f} MB，"
            f"上限: {config.audio_file.max_file_size_mb} MB"
        )
        return result

    # 4. 讀取 metadata
    try:
        metadata = get_audio_metadata(file_path)
    except Exception as e:
        result["error_message"] = f"無法讀取音檔: {e}"
        return result

    result["sample_rate"] = metadata["sample_rate"]
    result["duration"] = round(metadata["duration"], 2)
    result["channels"] = metadata["channels"]
    result["bit_depth"] = metadata["bit_depth"]

    # 5. 嚴格驗證模式
    if strict:
        # 檢查取樣率
        if metadata["sample_rate"] not in config.audio_file.sample_rates:
            errors.append(
                f"不支援的取樣率: {metadata['sample_rate']} Hz。"
                f"支援: {config.audio_file.sample_rates}"
            )

        # 檢查位元深度 (僅對 WAV 嚴格檢查)
        if suffix == 'wav' and metadata["bit_depth"] not in config.audio_file.bit_depths:
            warnings.append(
                f"非標準位元深度: {metadata['bit_depth']}-bit。"
                f"建議: {config.audio_file.bit_depths}"
            )

        # 檢查聲道數
        if metadata["channels"] > config.audio_file.max_channels:
            warnings.append(
                f"多聲道音檔 ({metadata['channels']} 聲道)，"
                f"將自動轉換為單聲道"
            )

        # 檢查長度
        min_dur, max_dur = config.audio_file.duration_range
        if metadata["duration"] < min_dur:
            errors.append(
                f"音檔太短: {metadata['duration']:.1f} 秒，"
                f"最少需要 {min_dur} 秒"
            )
        elif metadata["duration"] > max_dur:
            errors.append(
                f"音檔太長: {metadata['duration']:.1f} 秒，"
                f"上限 {max_dur} 秒"
            )

    # 6. 彙整結果
    result["warnings"] = warnings

    if errors:
        result["error_message"] = "; ".join(errors)
        result["file_valid"] = False
    else:
        result["file_valid"] = True

    return result


def load_and_validate(
    file_path: Union[str, Path],
    target_sr: Optional[int] = None
) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """載入並驗證音檔 (整合函數)

    一次完成驗證與載入，適合 API 使用。

    Args:
        file_path: 音檔路徑
        target_sr: 目標取樣率，None 則保留原始取樣率

    Returns:
        Tuple[np.ndarray, int, Dict]: (音訊資料, 取樣率, 驗證結果)

    Raises:
        AudioValidationError: 驗證失敗
        AudioLoadError: 載入失敗
    """
    # 先驗證
    validation = validate_audio(file_path, strict=True)

    if not validation["file_valid"]:
        raise AudioValidationError(validation["error_message"])

    # 載入音檔
    y, sr = load_audio(file_path, target_sr=target_sr, mono=True)

    return y, sr, validation


def load_audio_bytes(
    audio_bytes: bytes,
    file_format: str = "wav"
) -> Tuple[np.ndarray, int]:
    """從 bytes 載入音檔

    適用於處理上傳的檔案內容。

    Args:
        audio_bytes: 音檔的 bytes 內容
        file_format: 檔案格式 (wav, mp3, flac)

    Returns:
        Tuple[np.ndarray, int]: (音訊資料, 取樣率)
    """
    import io
    import tempfile

    # 建立臨時檔案
    with tempfile.NamedTemporaryFile(
        suffix=f".{file_format}",
        delete=False
    ) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

    try:
        y, sr = load_audio(tmp_path, mono=True)
        return y, sr
    finally:
        # 清理臨時檔案
        os.unlink(tmp_path)
