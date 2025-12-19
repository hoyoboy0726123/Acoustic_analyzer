# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - 音頻分析資料模型

此檔案定義所有音頻分析相關的 Pydantic 資料模型，
用於 API 請求/回應的資料驗證與序列化。
"""

from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field


class AudioFileInfo(BaseModel):
    """音檔資訊

    用於回傳音檔上傳後的驗證結果與 metadata。
    """
    file_valid: bool = Field(..., description="檔案是否有效")
    sample_rate: int = Field(..., description="取樣率 (Hz)")
    duration: float = Field(..., description="音檔長度 (秒)")
    channels: int = Field(..., description="聲道數")
    bit_depth: int = Field(..., description="位元深度")
    error_message: Optional[str] = Field(None, description="錯誤訊息")


class FFTResult(BaseModel):
    """FFT 頻譜分析結果"""
    frequencies: List[float] = Field(..., description="頻率陣列 (Hz)")
    magnitudes: List[float] = Field(..., description="對應能量值 (dB)")
    spectrum_image: Optional[str] = Field(None, description="頻譜圖 Base64 編碼")


class NoiseLevelResult(BaseModel):
    """噪音等級計算結果 (dB(A))"""
    leq_dba: float = Field(..., description="等效連續音壓級")
    lmax_dba: float = Field(..., description="最大音壓級")
    lmin_dba: float = Field(..., description="最小音壓級")
    l10: float = Field(..., description="L10 統計音壓級")
    l90: float = Field(..., description="L90 統計音壓級")


class ToneInfo(BaseModel):
    """單一 Discrete Tone 資訊"""
    frequency: float = Field(..., description="頻率 (Hz)")
    prominence: float = Field(..., description="突出量 (dB)")
    band: str = Field(..., description="所屬頻帶")


class DiscreteToneResult(BaseModel):
    """Discrete Tone 檢測結果"""
    tone_detected: bool = Field(..., description="是否偵測到 Discrete Tone")
    tones: List[ToneInfo] = Field(default_factory=list, description="偵測到的 Tone 列表")


class BandInfo(BaseModel):
    """單一頻帶分析資訊"""
    range: str = Field(..., description="頻率範圍")
    energy_db: float = Field(..., description="能量值 (dB)")
    status: Literal["PASS", "WARNING", "FAIL"] = Field(..., description="狀態")


class HighFreqPeak(BaseModel):
    """高頻峰值資訊"""
    frequency: float = Field(..., description="頻率 (Hz)")
    magnitude_db: float = Field(..., description="幅度 (dB)")
    prominence_db: float = Field(..., description="突出量 (dB)")


class BandAnalysis(BaseModel):
    """頻帶分離分析結果"""
    low_freq: BandInfo = Field(..., description="低頻帶分析")
    mid_freq: BandInfo = Field(..., description="中頻帶分析")
    mid_high_freq: BandInfo = Field(..., description="中高頻帶分析")
    high_freq: BandInfo = Field(..., description="高頻帶分析")
    ultra_high_freq: BandInfo = Field(..., description="超高頻帶分析")


class HighFreqResult(BaseModel):
    """高頻音隔離分析結果"""
    band_analysis: BandAnalysis = Field(..., description="頻帶分離分析")
    high_freq_peaks: List[HighFreqPeak] = Field(
        default_factory=list, description="高頻峰值列表"
    )
    coil_whine_detected: bool = Field(..., description="是否偵測到電感嘯叫")
    coil_whine_frequency: Optional[float] = Field(None, description="電感嘯叫頻率 (Hz)")
    coil_whine_prominence: Optional[float] = Field(None, description="電感嘯叫突出量 (dB)")
    possible_cause: Optional[str] = Field(None, description="可能原因")
    recommendation: Optional[str] = Field(None, description="建議")
    overall_status: Literal["PASS", "WARNING", "FAIL"] = Field(..., description="整體狀態")
    filtered_spectrum_image: Optional[str] = Field(None, description="濾波後頻譜圖 Base64")


class AnalysisReport(BaseModel):
    """完整分析報告"""
    task_id: str = Field(..., description="分析任務 ID")
    status: Literal["pending", "processing", "completed", "failed"] = Field(
        ..., description="任務狀態"
    )
    audio_info: Optional[AudioFileInfo] = Field(None, description="音檔資訊")
    fft_result: Optional[FFTResult] = Field(None, description="FFT 分析結果")
    noise_level: Optional[NoiseLevelResult] = Field(None, description="噪音等級")
    discrete_tone: Optional[DiscreteToneResult] = Field(None, description="Discrete Tone")
    high_freq_analysis: Optional[HighFreqResult] = Field(None, description="高頻分析")
    overall_status: Literal["PASS", "WARNING", "FAIL"] = Field(..., description="整體狀態")
    report_text: Optional[str] = Field(None, description="文字報告")
