# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - ASUS SOP 高頻音 (SPL) 分析模組

此模組實作 ASUS SPL 高頻音 SOP Ver.1.6.6.1 的判定邏輯：
1. IDLE Mode: 10s Moving Leq, 1s Shift, Max value analysis.
2. UE Mode (Boot): Single Leq for total duration.
3. Workload Mode: 1s Moving Leq, 0.5s Shift, Fail rate & Max value analysis.
"""

import numpy as np
from typing import Dict, Any, Tuple, List
from core.noise_level import apply_a_weighting, calculate_rms, rms_to_db, calculate_leq, REFERENCE_PRESSURE

def calculate_moving_leq(
    audio: np.ndarray, 
    sample_rate: int, 
    window_sec: float, 
    shift_sec: float
) -> Tuple[np.ndarray, np.ndarray]:
    """計算滑動平均 Leq (Moving Leq)
    
    Args:
        audio: 音訊資料 (1D array)
        sample_rate: 取樣率
        window_sec: 視窗長度 (秒, Delta t)
        shift_sec: 移動步長 (秒, Shift)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (時間軸, Leq值陣列)
    """
    window_samples = int(window_sec * sample_rate)
    shift_samples = int(shift_sec * sample_rate)
    
    # 先套用 A-weighting
    audio_weighted = apply_a_weighting(audio, sample_rate)
    
    n_samples = len(audio_weighted)
    if n_samples < window_samples:
        # 太短則回傳整段 Leq
        single_leq = calculate_leq(np.array([rms_to_db(calculate_rms(audio_weighted), REFERENCE_PRESSURE)]))
        return np.array([0]), np.array([single_leq])
    
    # 計算框數
    n_frames = (n_samples - window_samples) // shift_samples + 1
    leq_values = np.zeros(n_frames)
    times = np.zeros(n_frames)
    
    # 預先計算平方以加速
    audio_sq = audio_weighted ** 2
    
    for i in range(n_frames):
        start = i * shift_samples
        end = start + window_samples
        # 計算區間 RMS
        rms = np.sqrt(np.mean(audio_sq[start:end]))
        leq_values[i] = rms_to_db(rms, REFERENCE_PRESSURE)
        times[i] = start / sample_rate
        
    return times, leq_values

def analyze_idle_mode(audio: np.ndarray, sample_rate: int, spec_limit: float = 20.0) -> Dict[str, Any]:
    """執行 IDLE Mode 分析 (10s Window, 1s Shift)
    
    判定標準: 
    1. Max Leq 值為最終結果
    2. 所有點須 under SPEC 管制線
    """
    times, leqs = calculate_moving_leq(audio, sample_rate, window_sec=10.0, shift_sec=1.0)
    
    max_val = np.max(leqs)
    all_under = np.all(leqs <= spec_limit)
    
    return {
        "mode": "IDLE",
        "times": times,
        "leqs": leqs,
        "max_leq": float(round(max_val, 2)),
        "spec_limit": spec_limit,
        "is_pass": bool(all_under),
        "total_points": len(leqs),
        "summary": f"IDLE 分析完成。Max: {max_val:.2f} dBA, 管制線: {spec_limit} dBA, 結果: {'PASS' if all_under else 'FAIL'}"
    }

def analyze_ue_mode(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """執行 UE Mode 分析 (整段平均 Leq)
    
    判定標準: 整段錄音的平均聲壓值
    """
    audio_weighted = apply_a_weighting(audio, sample_rate)
    rms = calculate_rms(audio_weighted)
    leq = rms_to_db(rms, REFERENCE_PRESSURE)
    
    return {
        "mode": "UE",
        "leq": float(round(leq, 2)),
        "duration": len(audio) / sample_rate,
        "summary": f"UE (開機) 分析完成。整段平均: {leq:.2f} dBA"
    }

def analyze_workload_mode(
    audio: np.ndarray, 
    sample_rate: int, 
    spec_limit_fail_rate: float = 22.0,
    spec_limit_max: float = 28.0
) -> Dict[str, Any]:
    """執行 Workload Mode 分析 (1s Window, 0.5s Shift)
    
    判定標準:
    1. 統計超過 22 dBA 的點數比例 (Fail rate < 2%)
    2. Max 值 < 28 dBA
    """
    times, leqs = calculate_moving_leq(audio, sample_rate, window_sec=1.0, shift_sec=0.5)
    
    max_val = np.max(leqs)
    
    # 判斷超過 22 dBA 的點
    exceed_count = np.sum(leqs > spec_limit_fail_rate)
    fail_rate = (exceed_count / len(leqs)) * 100
    
    # 判定 1: Fail rate < 2% (SOP 提到 239 點允許 4 點以下)
    pass_rate = fail_rate < 2.0
    # 判定 2: Max < 28 dBA
    pass_max = max_val < spec_limit_max
    
    is_pass = pass_rate and pass_max
    
    return {
        "mode": "Workload",
        "times": times,
        "leqs": leqs,
        "max_leq": float(round(max_val, 2)),
        "fail_rate": float(round(fail_rate, 2)),
        "exceed_count": int(exceed_count),
        "spec_limit_rate": spec_limit_fail_rate,
        "spec_limit_max": spec_limit_max,
        "is_pass": bool(is_pass),
        "criteria_rate_pass": bool(pass_rate),
        "criteria_max_pass": bool(pass_max),
        "summary": f"Workload 分析完成。Max: {max_val:.2f} dBA, Fail Rate: {fail_rate:.1f}%, 結果: {'PASS' if is_pass else 'FAIL'}"
    }
