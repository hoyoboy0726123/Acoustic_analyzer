# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - Streamlit Web UI

功能 (AUD-009):
- 檔案上傳介面
- 分析結果顯示
- 頻譜圖視覺化
- 報告下載
"""

import streamlit as st
import tempfile
import os
from pathlib import Path
import sys

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Streamlit 應用程式主函數"""
    st.set_page_config(
        page_title="聲學測試 AI 分析系統",
        page_icon="🔊",
        layout="wide"
    )
    
    # 初始化 session_state
    if 'audio_loaded' not in st.session_state:
        st.session_state.audio_loaded = False
    if 'audio_original' not in st.session_state:
        st.session_state.audio_original = None
    if 'sr' not in st.session_state:
        st.session_state.sr = None
    if 'validation' not in st.session_state:
        st.session_state.validation = None

    st.title("🔊 聲學測試 AI 分析系統")
    st.markdown("*基於 AI 的筆記型電腦聲學測試分析系統*")
    st.markdown("---")

    # 側邊欄設定
    with st.sidebar:
        st.header("⚙️ 分析設定")
        
        highpass_cutoff = st.slider(
            "高通濾波截止頻率 (Hz)",
            min_value=1000, max_value=8000, value=4000, step=500,
            help="用於高頻音隔離分析"
        )
        
        st.markdown("---")
        
        st.subheader("📋 分析選項")
        analyze_noise = st.checkbox("噪音等級分析 dB(A)", value=True)
        analyze_spectrum = st.checkbox("FFT 頻譜分析", value=True)
        analyze_discrete_tone = st.checkbox("Discrete Tone 檢測", value=True)
        analyze_high_freq = st.checkbox("高頻音隔離分析", value=True)
        analyze_band_filter = st.checkbox("🎚️ 頻帶過濾模擬", value=False)
        
        # 頻帶選擇器
        removed_bands = []
        if analyze_band_filter:
            st.markdown("---")
            st.subheader("🎚️ 頻帶過濾模擬")
            st.caption("選擇要移除的頻帶，模擬去除特定噪音來源的效果")
            
            remove_low = st.checkbox("移除低頻 (20-500Hz) - 風扇/馬達", value=False, key="rm_low")
            remove_mid = st.checkbox("移除中頻 (500-2kHz) - 機械運轉", value=False, key="rm_mid")
            remove_mid_high = st.checkbox("移除中高頻 (2-6kHz) - 鍵盤聲", value=False, key="rm_mid_high")
            remove_high = st.checkbox("移除高頻 (6-12kHz) - 電感嘯叫", value=False, key="rm_high")
            remove_ultra = st.checkbox("移除超高頻 (12-20kHz)", value=False, key="rm_ultra")
            
            if remove_low:
                removed_bands.append("low_freq")
            if remove_mid:
                removed_bands.append("mid_freq")
            if remove_mid_high:
                removed_bands.append("mid_high_freq")
            if remove_high:
                removed_bands.append("high_freq")
            if remove_ultra:
                removed_bands.append("ultra_high_freq")
        
        st.markdown("---")
        st.caption("v1.0.0 | 聲學測試 AI 分析系統")

    # 主要內容區
    st.header("📁 上傳音檔")
    
    uploaded_file = st.file_uploader(
        "選擇要分析的音檔",
        type=["wav", "mp3", "flac"],
        help="支援 WAV、MP3、FLAC 格式，檔案大小上限 50MB"
    )

    if uploaded_file is not None:
        st.success(f"✅ 已上傳: **{uploaded_file.name}** ({uploaded_file.size / 1024 / 1024:.2f} MB)")
        
        # 開始分析按鈕 - 只載入音檔一次
        if st.button("🚀 開始分析", type="primary", use_container_width=True):
            load_audio_file(uploaded_file)
        
        # 如果音檔已載入，根據側邊欄設定即時顯示分析結果
        if st.session_state.audio_loaded:
            render_analysis_results(
                highpass_cutoff,
                analyze_noise,
                analyze_spectrum,
                analyze_discrete_tone,
                analyze_high_freq,
                analyze_band_filter,
                removed_bands
            )
    else:
        # 清除已載入的音檔
        st.session_state.audio_loaded = False
        st.session_state.audio_original = None
        
        st.info("👆 請上傳音檔以開始分析")
        
        # 顯示支援的規格
        with st.expander("📌 支援的音檔規格"):
            st.markdown("""
            | 項目 | 規格 |
            |------|------|
            | 格式 | WAV (必須), MP3, FLAC (可選) |
            | 取樣率 | 44100 或 48000 Hz |
            | 位元深度 | 16-bit 或 24-bit |
            | 聲道 | Mono (單聲道) |
            | 檔案大小 | ≤ 50 MB |
            | 長度 | 10 - 120 秒 |
            """)


def load_audio_file(uploaded_file):
    """載入音檔到 session_state"""
    with st.spinner("🔄 正在載入並驗證音檔..."):
        from core.audio_loader import load_audio, validate_audio
        
        # 建立臨時檔案
        with tempfile.NamedTemporaryFile(
            suffix=f".{uploaded_file.name.split('.')[-1]}",
            delete=False
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            # 驗證音檔
            validation = validate_audio(tmp_path, strict=False)
            
            if not validation["file_valid"]:
                st.error(f"❌ 音檔驗證失敗: {validation['error_message']}")
                return
            
            # 載入音檔
            audio, sr = load_audio(tmp_path)
            
            # 保存到 session_state
            st.session_state.audio_original = audio
            st.session_state.sr = sr
            st.session_state.validation = validation
            st.session_state.audio_loaded = True
            
            st.rerun()  # 重新運行以顯示分析結果
            
        finally:
            # 清理臨時檔案
            os.unlink(tmp_path)


def render_analysis_results(highpass_cutoff, analyze_noise, analyze_spectrum, 
                            analyze_discrete_tone, analyze_high_freq, 
                            analyze_band_filter, removed_bands):
    """根據側邊欄設定即時渲染分析結果"""
    import numpy as np
    
    # 從 session_state 取得音訊資料
    audio_original = st.session_state.audio_original
    sr = st.session_state.sr
    validation = st.session_state.validation
    
    if audio_original is None:
        return
    
    # 套用頻帶過濾 (如果啟用)
    if analyze_band_filter and removed_bands:
        with st.spinner("🎚️ 套用頻帶過濾..."):
            audio = apply_band_filter(audio_original, sr, removed_bands)
            st.info(f"🎚️ **頻帶過濾已啟用**: 已移除 {len(removed_bands)} 個頻帶，以下所有分析基於過濾後的音訊")
    else:
        audio = audio_original
    
    # 顯示音檔資訊
    display_audio_info(validation)
    
    # 執行各項分析 (使用過濾後的音訊)
    if analyze_noise:
        run_noise_analysis(audio, sr)

    if analyze_spectrum:
        run_spectrum_analysis(audio, sr)

    if analyze_discrete_tone:
        run_discrete_tone_analysis(audio, sr)

    if analyze_high_freq:
        run_high_freq_analysis(audio, sr, highpass_cutoff)

    # 如果有頻帶過濾，顯示原始 vs 過濾後對比
    if analyze_band_filter and removed_bands:
        run_band_filter_comparison(audio_original, audio, sr, removed_bands)

    st.success("✅ 分析完成！側邊欄調整設定會即時更新圖表。")



def display_audio_info(validation: dict):
    """顯示音檔資訊"""
    st.subheader("📊 音檔資訊")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("取樣率", f"{validation['sample_rate']} Hz")
    with col2:
        st.metric("長度", f"{validation['duration']:.2f} 秒")
    with col3:
        st.metric("位元深度", f"{validation['bit_depth']}-bit")
    with col4:
        st.metric("檔案大小", f"{validation['file_size_mb']:.2f} MB")
    
    if validation.get("warnings"):
        for warning in validation["warnings"]:
            st.warning(f"⚠️ {warning}")


def run_noise_analysis(audio, sr):
    """執行噪音等級分析"""
    from core.noise_level import calculate_noise_level
    
    result = calculate_noise_level(audio, sr)
    
    st.subheader("🔊 噪音等級分析 dB(A)")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Leq", f"{result['leq_dba']:.1f} dB")
    with col2:
        st.metric("Lmax", f"{result['lmax_dba']:.1f} dB")
    with col3:
        st.metric("Lmin", f"{result['lmin_dba']:.1f} dB")
    with col4:
        st.metric("L10", f"{result['l10']:.1f} dB")
    with col5:
        st.metric("L90", f"{result['l90']:.1f} dB")
    
    st.markdown("---")


def run_spectrum_analysis(audio, sr):
    """執行頻譜分析 - 多種圖表即時切換"""
    from core.fft import compute_average_spectrum
    from utils.interactive_plots import (
        create_interactive_spectrum,
        create_waveform_chart,
        create_spectrogram_chart,
        create_a_weighting_chart,
        create_octave_band_chart,
        create_waterfall_3d_chart,
        create_combined_analysis_chart
    )
    import numpy as np
    
    # 計算頻譜 (只需計算一次)
    frequencies, magnitudes_db = compute_average_spectrum(audio, sr)
    
    # 將結果存入 session_state 供圖表切換使用
    st.session_state['audio'] = audio
    st.session_state['sr'] = sr
    st.session_state['frequencies'] = frequencies
    st.session_state['magnitudes_db'] = magnitudes_db
    
    st.subheader("📈 頻譜分析 (多種視圖)")
    st.caption("💡 提示: 切換不同圖表類型即時顯示，支援滑鼠縮放、平移、十字座標")
    
    # 使用 tabs 實現即時切換
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 FFT 頻譜", 
        "🌊 波形圖", 
        "🔥 Spectrogram", 
        "👂 A-weighting", 
        "📶 1/3 倍頻程",
        "🌀 3D Waterfall",
        "📋 綜合視圖"
    ])
    
    with tab1:
        spectrum_fig = create_interactive_spectrum(
            frequencies, magnitudes_db,
            title="FFT 平均頻譜圖"
        )
        st.plotly_chart(spectrum_fig, use_container_width=True, key="fft_spectrum")
    
    with tab2:
        waveform_fig = create_waveform_chart(audio, sr)
        st.plotly_chart(waveform_fig, use_container_width=True, key="waveform")
    
    with tab3:
        spectrogram_fig = create_spectrogram_chart(audio, sr)
        st.plotly_chart(spectrogram_fig, use_container_width=True, key="spectrogram")
    
    with tab4:
        a_weight_fig = create_a_weighting_chart(sr)
        st.plotly_chart(a_weight_fig, use_container_width=True, key="a_weight")
        st.info("💡 A-weighting 曲線顯示人耳對不同頻率的敏感度。低頻和超高頻會被衰減，2-5kHz 區域（人耳最敏感）則接近 0 dB。")
    
    with tab5:
        octave_fig = create_octave_band_chart(audio, sr)
        st.plotly_chart(octave_fig, use_container_width=True, key="octave")
        st.info("💡 1/3 倍頻程分析依 ISO 標準將頻譜分成標準頻帶，常用於噪音評估和聲學測量。")
    
    with tab6:
        waterfall_fig = create_waterfall_3d_chart(audio, sr)
        st.plotly_chart(waterfall_fig, use_container_width=True, key="waterfall")
        st.info("💡 3D Waterfall 圖可旋轉、縮放。拖曳可改變視角，滾輪縮放。")
    
    with tab7:
        combined_fig = create_combined_analysis_chart(audio, sr, frequencies, magnitudes_db)
        st.plotly_chart(combined_fig, use_container_width=True, key="combined")
    
    st.markdown("---")


def run_discrete_tone_analysis(audio, sr):
    """執行 Discrete Tone 檢測"""
    from core.discrete_tone import detect_discrete_tones
    from core.fft import compute_average_spectrum, get_frequency_range
    from utils.interactive_plots import create_discrete_tone_chart
    
    result = detect_discrete_tones(audio, sr)
    
    st.subheader("🎵 Discrete Tone 檢測 (ECMA-74)")
    st.caption("💡 提示: 紅色星號標記超過門檻的 Discrete Tone，灰色三角形為候選峰值")
    
    # 計算頻譜用於視覺化
    frequencies, magnitudes_db = compute_average_spectrum(audio, sr)
    frequencies, magnitudes_db = get_frequency_range(frequencies, magnitudes_db, 50, 15000)
    
    # 繪製 Discrete Tone 視覺化圖表
    tone_fig = create_discrete_tone_chart(
        frequencies, magnitudes_db,
        tones=result.get("tones", []),
        all_candidates=result.get("all_candidates", []),
        title="Discrete Tone 檢測結果 (ECMA-74 標準)"
    )
    st.plotly_chart(tone_fig, use_container_width=True)
    
    # 狀態顯示
    if result["tone_detected"]:
        st.warning(f"⚠️ 偵測到 {len(result['tones'])} 個 Discrete Tone!")
        
        # 顯示偵測到的 Tone
        for i, tone in enumerate(result["tones"], 1):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(f"Tone #{i} 頻率", f"{tone['frequency']:.0f} Hz")
            with col2:
                st.metric("突出量", f"{tone['prominence']:.1f} dB")
            with col3:
                st.metric("門檻", f"{tone['threshold']:.0f} dB")
            with col4:
                st.metric("頻帶", tone['band'])
    else:
        st.success("✅ 未偵測到超過門檻的 Discrete Tone")
    
    # 顯示候選 Tone
    if result.get("all_candidates"):
        with st.expander("🔍 查看所有候選峰值"):
            import pandas as pd
            df = pd.DataFrame(result["all_candidates"])
            if not df.empty:
                df.columns = ["頻率 (Hz)", "突出量 (dB)", "幅度 (dB)", "頻帶", "門檻 (dB)", "超過門檻"]
                st.dataframe(df, use_container_width=True)
    
    st.markdown("---")


def run_high_freq_analysis(audio, sr, cutoff):
    """執行高頻音隔離分析"""
    from core.high_freq_detector import analyze_high_frequency
    from core.fft import compute_average_spectrum, get_frequency_range
    from core.filters import highpass_filter
    from utils.interactive_plots import (
        create_comparison_spectrum,
        create_dual_spectrum_comparison,
        create_band_energy_chart
    )
    import numpy as np
    
    result = analyze_high_frequency(audio, sr, cutoff)
    
    st.subheader("⚡ 高頻音隔離分析")
    
    # 整體狀態
    status = result["overall_status"]
    status_colors = {"PASS": "green", "WARNING": "orange", "FAIL": "red"}
    status_icons = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌"}
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("整體狀態", f"{status_icons[status]} {status}")
    with col2:
        st.metric("截止頻率", f"{cutoff} Hz")
    with col3:
        coil_status = "偵測到" if result["coil_whine_detected"] else "未偵測"
        st.metric("電感嘯叫", coil_status)
    
    # 電感嘯叫檢測結果
    if result["coil_whine_detected"]:
        st.error(
            f"🔔 **偵測到電感嘯叫!**\n\n"
            f"- 頻率: {result['coil_whine_frequency']:.0f} Hz\n"
            f"- 突出量: {result['coil_whine_prominence']:.1f} dB\n"
            f"- 可能原因: {result['possible_cause']}"
        )
    
    # 建議
    st.info(f"💡 **建議:** {result['recommendation']}")
    
    # === 濾波前後對比圖 (互動式) ===
    st.subheader("📊 濾波前後頻譜對比 (可縮放)")
    st.caption("💡 提示: 滑鼠滾輪縮放、拖曳平移、雙擊重置、滑鼠移動顯示十字座標")
    
    # 計算原始頻譜
    freqs_orig, mags_orig = compute_average_spectrum(audio, sr)
    
    # 計算濾波後頻譜
    audio_filtered = highpass_filter(audio, sr, cutoff)
    freqs_filt, mags_filt = compute_average_spectrum(audio_filtered, sr)
    
    # 限制顯示範圍
    freq_min, freq_max = 20, min(20000, sr // 2)
    freqs_orig, mags_orig = get_frequency_range(freqs_orig, mags_orig, freq_min, freq_max)
    freqs_filt, mags_filt = get_frequency_range(freqs_filt, mags_filt, freq_min, freq_max)
    
    # 繪製互動式對比圖
    comparison_fig = create_comparison_spectrum(
        freqs_orig, mags_orig, mags_filt, cutoff,
        title=f"高通濾波前後對比 (截止頻率: {cutoff} Hz)"
    )
    st.plotly_chart(comparison_fig, use_container_width=True)
    
    # 分開顯示原始和濾波後頻譜 (互動式雙欄)
    st.subheader("📈 原始 vs 濾波後頻譜對比")
    dual_fig = create_dual_spectrum_comparison(
        freqs_orig, mags_orig,
        freqs_filt, mags_filt,
        title1="原始頻譜 (全頻帶)",
        title2=f"高通濾波後 (>{cutoff} Hz)"
    )
    st.plotly_chart(dual_fig, use_container_width=True)
    
    # 頻帶能量分析圖 (互動式)
    st.subheader("📊 頻帶能量分析")
    if result.get("band_analysis"):
        band_fig = create_band_energy_chart(result["band_analysis"])
        st.plotly_chart(band_fig, use_container_width=True)
    
    # 高頻峰值列表
    if result.get("high_freq_peaks"):
        with st.expander("🔍 查看高頻峰值詳情"):
            import pandas as pd
            peaks_df = pd.DataFrame(result["high_freq_peaks"])
            if not peaks_df.empty:
                peaks_df.columns = ["頻率 (Hz)", "幅度 (dB)", "突出量 (dB)"]
                st.dataframe(peaks_df, use_container_width=True)
            else:
                st.info("無明顯高頻峰值")
    
    st.markdown("---")


def run_band_filter_analysis(audio, sr, removed_bands):
    """執行頻帶過濾模擬分析"""
    from core.fft import compute_average_spectrum, get_frequency_range
    from core.filters import bandpass_filter
    from core.noise_level import calculate_rms, rms_to_db
    from utils.interactive_plots import create_band_filter_comparison
    import numpy as np
    
    st.subheader("🎚️ 頻帶過濾模擬")
    st.caption("模擬移除特定頻帶後的頻譜變化")
    
    # 頻帶定義
    band_ranges = {
        'low_freq': (20, 500),
        'mid_freq': (500, 2000),
        'mid_high_freq': (2000, 6000),
        'high_freq': (6000, 12000),
        'ultra_high_freq': (12000, 20000)
    }
    
    band_names = {
        'low_freq': '低頻 (風扇/馬達)',
        'mid_freq': '中頻 (機械運轉)',
        'mid_high_freq': '中高頻 (鍵盤聲)',
        'high_freq': '高頻 (電感嘯叫)',
        'ultra_high_freq': '超高頻'
    }
    
    # 顯示移除的頻帶
    st.info(f"🔇 已移除的頻帶: {', '.join([band_names.get(b, b) for b in removed_bands])}")
    
    # 計算原始頻譜
    freqs_orig, mags_orig = compute_average_spectrum(audio, sr)
    
    # 建立過濾後的訊號 (通過保留未被移除的頻帶)
    nyquist = sr / 2
    audio_filtered = np.zeros_like(audio)
    
    for band_name, (low, high) in band_ranges.items():
        if band_name not in removed_bands:
            # 確保頻率在有效範圍內
            low = max(20, low)
            high = min(high, nyquist - 1)
            if low < high:
                try:
                    band_audio = bandpass_filter(audio, sr, low, high)
                    audio_filtered += band_audio
                except:
                    pass
    
    # 計算過濾後頻譜
    freqs_filt, mags_filt = compute_average_spectrum(audio_filtered, sr)
    
    # 限制顯示範圍
    freq_min, freq_max = 20, min(20000, sr // 2)
    freqs_orig, mags_orig = get_frequency_range(freqs_orig, mags_orig, freq_min, freq_max)
    freqs_filt, mags_filt = get_frequency_range(freqs_filt, mags_filt, freq_min, freq_max)
    
    # 繪製互動式對比圖
    filter_fig = create_band_filter_comparison(
        freqs_orig, mags_orig, mags_filt, removed_bands,
        title="頻帶過濾前後對比"
    )
    st.plotly_chart(filter_fig, use_container_width=True)
    
    # 計算能量變化
    col1, col2, col3 = st.columns(3)
    
    rms_orig = calculate_rms(audio)
    rms_filt = calculate_rms(audio_filtered)
    db_orig = rms_to_db(rms_orig, 1.0)
    db_filt = rms_to_db(rms_filt, 1.0)
    db_reduction = db_orig - db_filt
    
    with col1:
        st.metric("原始能量", f"{db_orig:.1f} dB")
    with col2:
        st.metric("過濾後能量", f"{db_filt:.1f} dB")
    with col3:
        st.metric("能量降低", f"{db_reduction:.1f} dB", delta=f"-{db_reduction:.1f}")
    
    st.markdown("---")


def apply_band_filter(audio, sr, removed_bands):
    """套用頻帶過濾，移除指定頻帶
    
    Args:
        audio: 原始音訊資料
        sr: 取樣率
        removed_bands: 要移除的頻帶列表
    
    Returns:
        過濾後的音訊資料
    """
    from core.filters import bandpass_filter
    import numpy as np
    
    # 頻帶定義
    band_ranges = {
        'low_freq': (20, 500),
        'mid_freq': (500, 2000),
        'mid_high_freq': (2000, 6000),
        'high_freq': (6000, 12000),
        'ultra_high_freq': (12000, 20000)
    }
    
    nyquist = sr / 2
    audio_filtered = np.zeros_like(audio)
    
    # 只保留未被移除的頻帶
    for band_name, (low, high) in band_ranges.items():
        if band_name not in removed_bands:
            low = max(20, low)
            high = min(high, nyquist - 1)
            if low < high:
                try:
                    band_audio = bandpass_filter(audio, sr, low, high)
                    audio_filtered += band_audio
                except:
                    pass
    
    return audio_filtered


def run_band_filter_comparison(audio_original, audio_filtered, sr, removed_bands):
    """顯示原始與過濾後的頻譜對比"""
    from core.fft import compute_average_spectrum, get_frequency_range
    from core.noise_level import calculate_rms, rms_to_db
    from utils.interactive_plots import create_band_filter_comparison
    
    st.subheader("🎚️ 頻帶過濾效果對比")
    
    band_names = {
        'low_freq': '低頻 (風扇/馬達)',
        'mid_freq': '中頻 (機械運轉)',
        'mid_high_freq': '中高頻 (鍵盤聲)',
        'high_freq': '高頻 (電感嘯叫)',
        'ultra_high_freq': '超高頻'
    }
    
    st.info(f"🔇 已移除的頻帶: {', '.join([band_names.get(b, b) for b in removed_bands])}")
    
    # 計算原始和過濾後頻譜
    freqs_orig, mags_orig = compute_average_spectrum(audio_original, sr)
    freqs_filt, mags_filt = compute_average_spectrum(audio_filtered, sr)
    
    freq_min, freq_max = 20, min(20000, sr // 2)
    freqs_orig, mags_orig = get_frequency_range(freqs_orig, mags_orig, freq_min, freq_max)
    freqs_filt, mags_filt = get_frequency_range(freqs_filt, mags_filt, freq_min, freq_max)
    
    # 繪製對比圖
    filter_fig = create_band_filter_comparison(
        freqs_orig, mags_orig, mags_filt, removed_bands,
        title="原始頻譜 vs 過濾後頻譜"
    )
    st.plotly_chart(filter_fig, use_container_width=True)
    
    # 能量變化
    col1, col2, col3 = st.columns(3)
    
    rms_orig = calculate_rms(audio_original)
    rms_filt = calculate_rms(audio_filtered)
    db_orig = rms_to_db(rms_orig, 1.0)
    db_filt = rms_to_db(rms_filt, 1.0)
    db_reduction = db_orig - db_filt
    
    with col1:
        st.metric("原始能量", f"{db_orig:.1f} dB")
    with col2:
        st.metric("過濾後能量", f"{db_filt:.1f} dB")
    with col3:
        st.metric("能量降低", f"{db_reduction:.1f} dB", delta=f"-{db_reduction:.1f}")
    
    st.markdown("---")


if __name__ == "__main__":
    main()


