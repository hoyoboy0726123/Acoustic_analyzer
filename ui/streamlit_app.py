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

from datetime import datetime
from utils.report import generate_excel_report


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
        
        # === HEAD ArtemiS 對齊功能 ===
        st.subheader("📊 頻譜分析模式")
        spectrum_mode = st.selectbox(
            "分析模式",
            options=['average', 'peak_hold', 'psd'],
            format_func=lambda x: {
                'average': '📊 FFT Average (平均)',
                'peak_hold': '📈 FFT Peak Hold (峰值保持)',
                'psd': '📉 PSD (功率頻譜密度)'
            }.get(x, x),
            help="Average: 時間平均 | Peak Hold: 取最大值 | PSD: 功率歸一化到 1 Hz"
        )
        
        window_function = st.selectbox(
            "窗函數",
            options=['hann', 'hamming', 'blackman', 'flattop'],
            format_func=lambda x: {
                'hann': '🔔 Hann (通用)',
                'hamming': '🔷 Hamming (更佳旁瓣抑制)',
                'blackman': '⚫ Blackman (最佳旁瓣抑制)',
                'flattop': '⬜ Flat Top (幅度精確)'
            }.get(x, x),
            help="Hann: 95% 應用適用 | Blackman: 需極佳旁瓣抑制 | Flat Top: 幅度校準"
        )
        
        highpass_cutoff = st.slider(
            "高通濾波截止頻率 (Hz)",
            min_value=1000, max_value=8000, value=4000, step=500,
            help="用於高頻音隔離分析"
        )
        
        st.markdown("---")
        
        st.subheader("📋 分析選項")
        analyze_noise = st.checkbox("噪音等級分析 dB(A)", value=True)
        analyze_spectrum = st.checkbox("FFT 頻譜分析", value=True)
        
        # A-weighting 選項 (預設啟用)
        use_a_weighting = st.checkbox(
            "👂 套用 A-weighting 加權", 
            value=True,
            help="A-weighting 模擬人耳對不同頻率的敏感度，符合 IEC 61672-1 標準"
        )
        
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
        st.subheader("📄 報告生成 (AUD-008)")
        
        if st.session_state.get('audio_loaded', False):
            if st.button("📊 生成 Excel 報告", key="btn_gen_report", use_container_width=True):
                with st.spinner("正在彙整數據並生成報告..."):
                    # 使用原始未過濾音訊
                    report_data, error = generate_excel_report(
                        st.session_state.audio_original,
                        st.session_state.sr,
                        filename=st.session_state.get('audio_filename', "audio.wav")
                    )
                    
                    if error:
                        st.error(error)
                    else:
                        st.session_state['report_xlsx'] = report_data
                        st.success("✅ 報告生成成功")

            if 'report_xlsx' in st.session_state:
                st.download_button(
                    label="⬇️ 下載 Excel 報表",
                    data=st.session_state['report_xlsx'],
                    file_name=f"Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
        else:
            st.caption("請先上傳音檔以啟用報告功能")

        st.markdown("---")
        st.caption("v1.0.0 | 聲學測試 AI 分析系統")

    # 主要內容區
    st.header("📁 上傳音檔")
    
    uploaded_files = st.file_uploader(
        "選擇要分析的音檔 (支援多選)",
        type=["wav", "mp3", "flac"],
        accept_multiple_files=True,
        help="支援 WAV、MP3、FLAC 格式，檔案大小上限 50MB"
    )

    if uploaded_files:
        if len(uploaded_files) == 1:
            uploaded_file = uploaded_files[0]
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
                    removed_bands,
                    use_a_weighting,
                    spectrum_mode,
                    window_function
                )
        else:
            # 批次模式
            st.success(f"✅ 已上傳 **{len(uploaded_files)}** 個檔案，準備進行批次分析")
            if st.button(f"🚀 開始批次分析", type="primary", use_container_width=True):
                process_batch_analysis(uploaded_files)
            
            if st.session_state.get('batch_data'):
                render_batch_dashboard(
                    highpass_cutoff,
                    analyze_noise,
                    analyze_spectrum,
                    analyze_discrete_tone,
                    analyze_high_freq,
                    analyze_band_filter,
                    removed_bands,
                    use_a_weighting,
                    spectrum_mode,
                    window_function
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
            st.session_state.audio_filename = uploaded_file.name
            
            st.rerun()  # 重新運行以顯示分析結果
            
        finally:
            # 清理臨時檔案
            os.unlink(tmp_path)


def render_analysis_results(highpass_cutoff, analyze_noise, analyze_spectrum, 
                            analyze_discrete_tone, analyze_high_freq, 
                            analyze_band_filter, removed_bands, use_a_weighting=True,
                            spectrum_mode='average', window_function='hann'):
    """根據側邊欄設定即時渲染分析結果
    
    Args:
        highpass_cutoff: 高通濾波截止頻率
        analyze_noise: 是否分析噪音等級
        analyze_spectrum: 是否分析頻譜
        analyze_discrete_tone: 是否檢測 Discrete Tone
        analyze_high_freq: 是否分析高頻
        analyze_band_filter: 是否啟用頻帶過濾
        removed_bands: 要移除的頻帶列表
        use_a_weighting: 是否套用 A-weighting
        spectrum_mode: 頻譜分析模式 (average/peak_hold/psd)
        window_function: 窗函數 (hann/hamming/blackman/flattop)
    """
    import numpy as np
    import io
    import soundfile as sf
    
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
    
    # 顯示加權模式
    # 顯示加權模式 (已移除單純展示)
    pass
    
    # 顯示音檔資訊
    display_audio_info(validation)
    
    # === 同步音訊播放器 (帶 Spectrogram 進度線) ===
    from ui.audio_player import create_audio_player_with_spectrogram, create_simple_audio_player
    
    if analyze_band_filter and removed_bands:
        # 有頻帶過濾時顯示兩個播放器
        col1, col2 = st.columns(2)
        with col1:
            st.caption("🎧 **過濾後音訊** (基於此進行分析)")
            create_audio_player_with_spectrogram(audio, sr, "🎵 過濾後音訊播放器")
        with col2:
            st.caption("🔊 **原始音訊** (對照參考)")
            create_audio_player_with_spectrogram(audio_original, sr, "🔊 原始音訊播放器")
    else:
        # 只顯示一個播放器
        create_audio_player_with_spectrogram(audio, sr, "🎵 音訊播放器 (點擊頻譜圖可跳轉)")
    
    st.markdown("---")
    
    # 執行各項分析 (使用過濾後的音訊，傳入 A-weighting 設定)
    if analyze_noise:
        run_noise_analysis(audio, sr)

    if analyze_spectrum:
        run_spectrum_analysis(audio, sr, use_a_weighting, spectrum_mode, window_function)

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


def run_spectrum_analysis(audio, sr, use_a_weighting=True, 
                          spectrum_mode='average', window_function='hann'):
    """執行頻譜分析 - 多種圖表即時切換
    
    Args:
        audio: 音訊資料
        sr: 取樣率
        use_a_weighting: 是否套用 A-weighting 加權
        spectrum_mode: 分析模式 (average/peak_hold/psd)
        window_function: 窗函數 (hann/hamming/blackman/flattop)
    """
    from core.fft import compute_spectrum_with_mode, apply_a_weighting
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
    
    # 分析模式對應的標籤
    mode_labels = {
        'average': 'FFT Average',
        'peak_hold': 'FFT Peak Hold',
        'psd': 'PSD'
    }
    mode_label = mode_labels.get(spectrum_mode, spectrum_mode)
    
    # 使用指定模式和窗函數計算頻譜
    frequencies, magnitudes_db, unit = compute_spectrum_with_mode(
        audio, sr, mode=spectrum_mode, window=window_function
    )
    
    # 套用 A-weighting (如果啟用)
    if use_a_weighting:
        magnitudes_db = apply_a_weighting(frequencies, magnitudes_db)
        weight_label = f"{unit}(A)" if unit != 'dB/Hz' else "dB(A)/Hz"
    else:
        weight_label = unit
    
    # 將結果存入 session_state 供圖表切換使用
    st.session_state['audio'] = audio
    st.session_state['sr'] = sr
    st.session_state['frequencies'] = frequencies
    st.session_state['magnitudes_db'] = magnitudes_db
    st.session_state['use_a_weighting'] = use_a_weighting
    st.session_state['spectrum_mode'] = spectrum_mode
    st.session_state['window_function'] = window_function
    
    st.subheader(f"📈 頻譜分析 [{mode_label}] - {weight_label}")
    st.caption(f"💡 模式: {mode_label} | 窗函數: {window_function.capitalize()} | 支援縮放、平移、十字座標")
    
    # 使用 tabs 實現即時切換
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        f"📊 FFT 頻譜 ({weight_label})", 
        "🌊 波形圖", 
        "🔥 Spectrogram", 
        f"📶 1/3 倍頻程 ({weight_label})",
        "🌀 3D Waterfall",
        "📋 綜合視圖"
    ])
    
    with tab1:
        spectrum_fig = create_interactive_spectrum(
            frequencies, magnitudes_db,
            title=f"FFT 平均頻譜圖 ({weight_label})",
            ylabel=f"幅度 ({weight_label})"
        )
        st.plotly_chart(spectrum_fig, use_container_width=True, key="fft_spectrum")
    
    with tab2:
        waveform_fig = create_waveform_chart(audio, sr)
        st.plotly_chart(waveform_fig, use_container_width=True, key="waveform")
    
    with tab3:
        spectrogram_fig = create_spectrogram_chart(audio, sr)
        st.plotly_chart(spectrogram_fig, use_container_width=True, key="spectrogram")
    
    with tab4:
        octave_fig = create_octave_band_chart(audio, sr, use_a_weighting=use_a_weighting)
        st.plotly_chart(octave_fig, use_container_width=True, key="octave")
        st.info("💡 1/3 倍頻程分析依 IEC 61260 標準，對齊 HEAD acoustics ArtemiS 計算方式。")
    
    with tab5:
        waterfall_fig = create_waterfall_3d_chart(audio, sr)
        st.plotly_chart(waterfall_fig, use_container_width=True, key="waterfall")
        st.info("💡 3D Waterfall 圖可旋轉、縮放。拖曳可改變視角，滾輪縮放。")
    
    with tab6:
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


def process_batch_analysis(uploaded_files):
    """執行批次分析"""
    import pandas as pd
    import tempfile
    import os
    from core.audio_loader import load_audio, validate_audio
    from core.noise_level import calculate_noise_level
    from core.fft import compute_average_spectrum
    from core.high_freq_detector import analyze_high_frequency
    from core.band_analyzer import compute_octave_bands
    
    batch_results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    count = len(uploaded_files)
    
    for i, file in enumerate(uploaded_files):
        status_text.text(f"正在分析 ({i+1}/{count}): {file.name}...")
        
        # Save temp
        suffix = f".{file.name.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
            
        try:
            # 1. Validate Audio
            validation = validate_audio(tmp_path, strict=False)
            
            # 2. Load Audio
            audio, sr = load_audio(tmp_path)
            
            # 3. Noise Level
            noise = calculate_noise_level(audio, sr)
            
            # 4. High Freq
            hf = analyze_high_frequency(audio, sr)
            
            # 5. Spectrum
            freqs, mags = compute_average_spectrum(audio, sr)
            
            # 6. 1/3 Octave Bands
            octave = compute_octave_bands(audio, sr, use_a_weighting=True)
            
            # Store Result
            batch_results[file.name] = {
                "noise": noise,
                "high_freq": hf,
                "spectrum": {"freqs": freqs, "mags": mags},
                "octave": octave,
                "sr": sr,
                "duration": len(audio)/sr,
                "audio": audio, # Save raw audio
                "validation": validation # Save validation info
            }
            
        except Exception as e:
            st.error(f"分析 {file.name} 失敗: {e}")
            
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
            
        progress_bar.progress((i + 1) / count)
        
    st.session_state['batch_data'] = batch_results
    status_text.success("批次分析完成!")
    st.rerun()


def render_batch_dashboard(
    highpass_cutoff,
    analyze_noise,
    analyze_spectrum,
    analyze_discrete_tone,
    analyze_high_freq,
    analyze_band_filter,
    removed_bands,
    use_a_weighting,
    spectrum_mode,
    window_function
):
    """顯示批次分析儀表板"""
    import plotly.graph_objects as go
    
    data = st.session_state.get('batch_data', {})
    if not data:
        return

    st.header("📊 批次分析比較儀表板")
    
    # 1. Comparison Table
    st.subheader("1. 數據總表")
    table_rows = []
    
    for name, res in data.items():
        n = res['noise']
        hf = res['high_freq']
        table_rows.append({
            "Filename": name,
            "Leq (dBA)": n['leq_dba'],
            "Lmax": n['lmax_dba'],
            "L90": n['l90'],
            "Coil Whine": "YES" if hf['coil_whine_detected'] else "NO",
            "CW Freq": f"{hf.get('coil_whine_frequency', 0):.0f}" if hf['coil_whine_detected'] else "-",
            "CW Prom": f"{hf.get('coil_whine_prominence', 0):.1f}" if hf['coil_whine_detected'] else "-"
        })
    
    import pandas as pd
    df = pd.DataFrame(table_rows)
    st.dataframe(df, use_container_width=True)
    
    st.download_button(
        label="⬇️ 下載比較總表 (CSV)",
        data=df.to_csv(index=False).encode('utf-8-sig'),
        file_name=f"Batch_Summary_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )
    

    # File Selector for Comparison Charts
    st.subheader("2. 詳細比較分析")
    st.caption("建議選擇 2-3 個檔案進行詳細比較，以免畫面過於擁擠")
    selected_files = st.multiselect("選擇要比較的檔案", options=list(data.keys()), default=list(data.keys())[:2])
    
    if not selected_files:
        st.info("請選擇至少一個檔案進行比較")
        return

    # Import visualization tools
    from utils.interactive_plots import (
        create_spectrogram_chart,
        create_waterfall_3d_chart,
        create_octave_band_chart
    )
    
    # 1. 1/3 Octave Comparison (Grouped Bar)
    st.markdown("#### 1/3 倍頻程比較 (Grouped Bar)")
    fig_oct = go.Figure()
    
    for name in selected_files:
        oct_data = data[name]['octave']
        # Use Bar for grouped comparison
        fig_oct.add_trace(go.Bar(
            x=oct_data['nominal_freqs'],
            y=oct_data['band_levels'],
            name=name,
            opacity=0.8
        ))
        
    fig_oct.update_layout(
        title="1/3 倍頻程頻譜比較",
        xaxis_title="頻率 (Hz)",
        yaxis_title="音壓級 dB(A)",
        xaxis_type="log",
        barmode='group', # Grouped bars
        hovermode="x unified"
    )
    st.plotly_chart(fig_oct, use_container_width=True)

    # 2. FFT Comparison (Line)
    st.markdown("#### FFT 細部頻譜比較 (Overlay)")
    fig_fft = go.Figure()
    for name in selected_files:
        spec = data[name]['spectrum']
        mask = spec['freqs'] <= 20000
        x_vals = spec['freqs'][mask]
        y_vals = spec['mags'][mask]
        
        fig_fft.add_trace(go.Scatter(
            x=x_vals, 
            y=y_vals,
            name=name,
            mode='lines',
            line=dict(width=1)
        ))
    fig_fft.update_layout(
        title="FFT 平均頻譜比較",
        xaxis_title="頻率 (Hz)",
        yaxis_title="幅度 (dB)",
        hovermode="x unified",
        xaxis_type="log"
    )
    st.plotly_chart(fig_fft, use_container_width=True)
    
    # 3. Level vs Time
    st.markdown("#### 噪音等級趨勢 (Level vs Time)")
    fig_time = go.Figure()
    has_profile = False
    
    for name in selected_files:
        profile = data[name]['noise'].get('profile', {})
        if profile and 'times' in profile and 'levels' in profile:
            has_profile = True
            times = profile['times']
            levels = profile['levels']
            if len(times) > 5000:
                step = len(times) // 5000
                times = times[::step]
                levels = levels[::step]
            
            fig_time.add_trace(go.Scatter(
                x=times, 
                y=levels,
                name=name,
                mode='lines',
                line=dict(width=1.5)
            ))
            
    if has_profile:
        fig_time.update_layout(
            title="噪音等級趨勢比較 (Leq Profile)",
            xaxis_title="時間 (秒)",
            yaxis_title="音壓級 dB(A)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_time, use_container_width=True)

    # 4. Spectrogram Comparison (Side-by-side)
    st.markdown("#### Spectrogram 對照比較")
    cols = st.columns(len(selected_files))
    for i, name in enumerate(selected_files):
        with cols[i]:
            st.markdown(f"**{name}**")
            audio_data = data[name].get('audio', None)
            sr_data = data[name].get('sr', 48000)
            if audio_data is not None:
                # Reuse existing function
                fig_spec = create_spectrogram_chart(audio_data, sr_data, title=f"Spectrogram: {name}")
                st.plotly_chart(fig_spec, use_container_width=True, key=f"batch_spec_{i}")
            else:
                st.warning("無音訊數據")

    # 5. 3D Waterfall Comparison (Side-by-side)
    st.markdown("#### 3D Waterfall 對照比較")
    cols_water = st.columns(len(selected_files))
    for i, name in enumerate(selected_files):
        with cols_water[i]:
            st.markdown(f"**{name}**")
            audio_data = data[name].get('audio', None)
            sr_data = data[name].get('sr', 48000)
            if audio_data is not None:
                fig_water = create_waterfall_3d_chart(audio_data, sr_data)
                # Update title
                fig_water.update_layout(title=f"Waterfall: {name}")
                st.plotly_chart(fig_water, use_container_width=True, key=f"batch_water_{i}")
            else:
                st.warning("無音訊數據")

        
    # --- Detail Inspector ---
    st.markdown("---")
    st.header("🔍 單檔詳細分析檢視 (Detail Inspector)")
    
    detail_file = st.selectbox("選擇要查看詳細報告的檔案", options=["(請選擇)"] + list(data.keys()))
    
    if detail_file and detail_file != "(請選擇)":
        target_data = data[detail_file]
        
        # Inject data into global session state to simulate Single File Mode
        st.session_state.audio_loaded = True
        st.session_state.audio_original = target_data['audio']
        st.session_state.sr = target_data['sr']
        st.session_state.audio_filename = detail_file
        # Fix: Inject validation info
        if 'validation' in target_data:
            st.session_state.validation = target_data['validation']
        else:
            # Fallback if old data present in session (should not happen if re-run)
            st.session_state.validation = {
                "file_valid": True,
                "sample_rate": target_data['sr'],
                "duration": target_data['duration'],
                "channels": 1,
                "bit_depth": 16, # Assume 16
                "file_size_mb": 0,
                "warnings": []
            }
        
        st.info(f"正在顯示 **{detail_file}** 的詳細分析結果...")
        
        # Reuse the main analysis renderer
        # Ensure we capture current sidebar settings
        # We need to access the sidebar widget values. They are in 'main' scope...
        # But Streamlit widgets are global in session_state usually.
        # However, variables like 'highpass_cutoff' are passed as args.
        # We need to grab them from session_state or default?
        # Sidebar widgets were defined in 'main()'. They are local variables there.
        # WE CANNOT ACCESS 'highpass_cutoff' here easily unless we pass them or read session state keys.
        
        render_analysis_results(
            highpass_cutoff,
            analyze_noise,
            analyze_spectrum,
            analyze_discrete_tone,
            analyze_high_freq,
            analyze_band_filter,
            removed_bands,
            use_a_weighting,
            spectrum_mode,
            window_function
        )



if __name__ == "__main__":
    main()


