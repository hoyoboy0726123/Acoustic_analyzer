# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - 音頻分析 API 路由

此檔案定義所有音頻分析相關的 API 端點：
- POST /audio/upload - 上傳音檔
- POST /audio/analyze - 執行完整音頻分析
- POST /audio/analyze/spectrum - FFT 頻譜分析
- POST /audio/analyze/noise-level - 噪音等級計算
- POST /audio/analyze/discrete-tone - Discrete Tone 檢測
- POST /audio/analyze/high-freq - 高頻音隔離分析
- POST /audio/analyze/spectrogram - 頻譜瀑布圖生成
"""

import tempfile
import os
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from typing import Optional

# 導入核心模組
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.audio_loader import load_audio, validate_audio, get_audio_metadata
from core.fft import analyze_spectrum
from core.noise_level import calculate_noise_level
from core.discrete_tone import detect_discrete_tones
from core.high_freq_detector import analyze_high_frequency
from core.band_analyzer import analyze_frequency_bands
from utils.visualization import plot_spectrum, plot_spectrogram, image_to_base64
from utils.report import generate_full_report


router = APIRouter(prefix="/audio", tags=["音頻分析"])


async def save_upload_file_tmp(upload_file: UploadFile) -> str:
    """將上傳檔案存為臨時檔案"""
    suffix = Path(upload_file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await upload_file.read()
        tmp.write(content)
        return tmp.name


@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """上傳音檔端點

    接收使用者上傳的音訊檔案，驗證格式與品質。
    """
    tmp_path = await save_upload_file_tmp(file)

    try:
        # 驗證音檔
        validation = validate_audio(tmp_path, strict=False)

        return {
            "success": validation["file_valid"],
            "filename": file.filename,
            "validation": validation
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.unlink(tmp_path)


@router.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """執行完整音頻分析

    對上傳的音檔執行所有分析功能。
    """
    tmp_path = await save_upload_file_tmp(file)

    try:
        # 驗證音檔
        validation = validate_audio(tmp_path, strict=False)
        if not validation["file_valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"音檔驗證失敗: {validation['error_message']}"
            )

        # 載入音檔
        audio, sr = load_audio(tmp_path)

        # 執行各項分析
        noise_result = calculate_noise_level(audio, sr)
        fft_result = analyze_spectrum(audio, sr)
        tone_result = detect_discrete_tones(audio, sr)
        high_freq_result = analyze_high_frequency(audio, sr)
        band_result = analyze_frequency_bands(audio, sr)

        # 準備檔案資訊
        file_info = {
            "file_name": file.filename,
            "sample_rate": validation["sample_rate"],
            "duration": validation["duration"],
            "format": validation.get("format", "WAV"),
            "bit_depth": validation.get("bit_depth"),
            "channels": validation.get("channels", 1)
        }

        # 生成報告
        report = generate_full_report(
            file_info=file_info,
            noise_level=noise_result,
            fft_analysis=fft_result,
            discrete_tone=tone_result,
            high_freq_analysis=high_freq_result,
            band_analysis=band_result
        )

        return report

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@router.post("/analyze/spectrum")
async def analyze_spectrum_endpoint(file: UploadFile = File(...)):
    """FFT 頻譜分析"""
    tmp_path = await save_upload_file_tmp(file)

    try:
        audio, sr = load_audio(tmp_path)
        result = analyze_spectrum(audio, sr)

        # 生成頻譜圖
        import numpy as np
        spectrum_png = plot_spectrum(
            np.array(result["frequencies"]),
            np.array(result["magnitudes_db"]),
            title=f"頻譜圖 - {file.filename}"
        )

        result["spectrum_image_base64"] = image_to_base64(spectrum_png)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@router.post("/analyze/noise-level")
async def analyze_noise_level(file: UploadFile = File(...)):
    """噪音等級計算"""
    tmp_path = await save_upload_file_tmp(file)

    try:
        audio, sr = load_audio(tmp_path)
        result = calculate_noise_level(audio, sr)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@router.post("/analyze/discrete-tone")
async def analyze_discrete_tone(file: UploadFile = File(...)):
    """Discrete Tone 檢測"""
    tmp_path = await save_upload_file_tmp(file)

    try:
        audio, sr = load_audio(tmp_path)
        result = detect_discrete_tones(audio, sr)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@router.post("/analyze/high-freq")
async def analyze_high_freq(
    file: UploadFile = File(...),
    filter_cutoff: Optional[int] = Form(4000)
):
    """高頻音隔離分析"""
    tmp_path = await save_upload_file_tmp(file)

    try:
        audio, sr = load_audio(tmp_path)
        result = analyze_high_frequency(audio, sr, cutoff=filter_cutoff)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@router.post("/analyze/spectrogram")
async def generate_spectrogram_endpoint(file: UploadFile = File(...)):
    """頻譜瀑布圖生成"""
    tmp_path = await save_upload_file_tmp(file)

    try:
        audio, sr = load_audio(tmp_path)
        spectrogram_png = plot_spectrogram(
            audio, sr,
            title=f"Spectrogram - {file.filename}"
        )

        return Response(
            content=spectrogram_png,
            media_type="image/png"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@router.post("/analyze/band")
async def analyze_bands(file: UploadFile = File(...)):
    """頻帶分離分析"""
    tmp_path = await save_upload_file_tmp(file)

    try:
        audio, sr = load_audio(tmp_path)
        result = analyze_frequency_bands(audio, sr)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)
