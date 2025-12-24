# -*- coding: utf-8 -*-
"""
聲學測試分析系統 - Streamlit Web UI

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
from utils.pdf_report import generate_pdf_report


def main():
    """Streamlit 應用程式主函數"""
    st.set_page_config(
        page_title="聲學測試分析系統",
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

    st.title("🔊 聲學測試分析系統")
    st.markdown("*專業級筆記型電腦聲學測試分析系統*")
    st.markdown("---")

    # 側邊欄設定
    with st.sidebar:
        st.header("⚙️ 分析設定")
        
        # === 麥克風校準 ===
        with st.expander("🎤 麥克風校準", expanded=False):
            st.markdown("""
            **校準方法**：
            1. 使用 94 dB 或 114 dB 校準器錄製校準音
            2. 查看系統顯示的 Leq 值
            3. 輸入偏移值 = 已知值 - 系統顯示值
            """)
            calibration_offset = st.number_input(
                "校準偏移 (dB)",
                min_value=-50.0,
                max_value=50.0,
                value=0.0,
                step=0.1,
                help="此值會加到所有 dB 分析結果上。例如：校準器 94 dB，系統顯示 70 dB，則輸入 +24"
            )
            if calibration_offset != 0:
                st.info(f"📌 已套用校準偏移: **{calibration_offset:+.1f} dB**")
        
        # 將校準偏移存入 session_state
        st.session_state['calibration_offset'] = calibration_offset
        
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
        
        # FFT 點數選擇器 (頻率解析度控制)
        n_fft_options = {
            4096: "4096 (標準，~11.7 Hz)",
            8192: "8192 (精細，~5.9 Hz)",
            16384: "16384 (高精度，~2.9 Hz)",
            32768: "32768 (超高精度，~1.5 Hz)"
        }
        n_fft = st.selectbox(
            "FFT 點數 (頻率解析度)",
            options=list(n_fft_options.keys()),
            format_func=lambda x: n_fft_options[x],
            index=1,  # 預設 8192
            help="點數越高，頻率解析度越精確，但計算時間越長"
        )
        
        # 顯示實際頻率解析度
        freq_resolution = 48000 / n_fft  # 假設 48kHz 取樣率
        st.caption(f"📐 頻率解析度: **{freq_resolution:.2f} Hz**")
        
        highpass_cutoff = st.slider(
            "高通濾波截止頻率 (Hz)",
            min_value=20, max_value=8000, value=20, step=10,
            help="濾除低於此頻率的聲音。20 Hz = 人耳下限（保留完整頻譜）。當「頻帶過濾模擬」開啟時，此設定會被忽略。"
        )
        
        st.markdown("---")
        
        st.subheader("📋 分析選項")
        analyze_noise = st.checkbox("噪音等級分析 dB(A)", value=True)
        
        leq_settings = {'spec': None, 'tag': ''}
        if analyze_noise:
            col_tag, col_spec = st.columns([1, 1])
            with col_tag:
                leq_tag = st.text_input(
                    "測試標籤 (選填)",
                    placeholder="例如: IDLE",
                    help="測試情境標籤，將顯示於報告中"
                )
            with col_spec:
                leq_spec_val = st.number_input(
                    "Leq 標準 (dB)",
                    min_value=0.0,
                    max_value=120.0,
                    value=0.0,
                    step=1.0,
                    help="若測量 > 標準則顯示 FAIL"
                )
            
            if leq_spec_val > 0:
                leq_settings['spec'] = leq_spec_val
                leq_settings['tag'] = leq_tag if leq_tag else "Noise Check"
        analyze_spectrum = st.checkbox("FFT 頻譜分析", value=True)
        
        if analyze_spectrum:
            # Level vs Time 平滑設定
            smooth_window_size = st.number_input(
                "🌊 Level vs Time 平滑度 (Smoothing)",
                min_value=1,
                value=1,
                step=1,
                help="設定 Level vs Time 圖表的移動平均窗口大小。1 為原始數據 (最靈敏)，數值越大越平滑。"
            )
            st.caption(f"目前設定: {'原始數據 (Fast)' if smooth_window_size == 1 else f'平滑視窗 {smooth_window_size} frames'}")
        else:
            smooth_window_size = 1 # Default if hidden
        
        # A-weighting 選項 (預設啟用)
        use_a_weighting = st.checkbox(
            "👂 套用 A-weighting 加權", 
            value=True,
            help="A-weighting 模擬人耳對不同頻率的敏感度，符合 IEC 61672-1 標準"
        )
        
        # Spectrogram 顯示模式
        st.markdown("##### 🎨 Spectrogram 設定")
        

        
        # dB SPL 絕對模式
        spectrogram_use_spl = st.checkbox(
            "📊 dB SPL 絕對模式",
            value=False,
            help="啟用後顯示絕對 dB SPL 值（需要校準偏移）。預設使用相對功率 dB。"
        )
        
        # Spectrogram 校準偏移（只在 dB SPL 模式下顯示）
        if spectrogram_use_spl:
            spectrogram_spl_offset = st.number_input(
                "Spectrogram 校準偏移 (dB)",
                value=0.0,
                step=10.0,
                help="將相對 dB 轉換為 dB SPL 所需的偏移量。可從 HEAD acoustics 對比獲得。"
            )
            st.info("💡 對比 HEAD acoustics 的相同音訊來確定偏移值")
        else:
            spectrogram_spl_offset = 0.0
        
        # Spectrogram 色彩範圍控制
        spectrogram_auto_range = st.checkbox(
            "🔄 自動範圍",
            value=True,
            help="自動調整 Spectrogram 色彩範圍。關閉後可手動設定。"
        )
        
        if spectrogram_auto_range:
            spectrogram_z_range = None
        else:
            spec_col1, spec_col2 = st.columns(2)
            with spec_col1:
                spec_z_min = st.number_input("最小值 (dB)", value=-100, step=10)
            with spec_col2:
                spec_z_max = st.number_input("最大值 (dB)", value=-60, step=10)
            spectrogram_z_range = (spec_z_min, spec_z_max)
        
        analyze_discrete_tone = st.checkbox("Discrete Tone 檢測", value=True)
        
        # ECMA 標準選擇
        ecma_standard = 'ECMA-74'
        if analyze_discrete_tone:
            ecma_standard = st.radio(
                "Discrete Tone 判定標準",
                options=['ECMA-74', 'ECMA-418'],
                index=0,  # 預設 ECMA-74
                horizontal=True,
                help="ECMA-74: 固定頻帶門檻 (較寬鬆) | ECMA-418: 公式計算門檻 (較嚴格)"
            )
        
        analyze_sop = st.checkbox("ASUS SOW 高頻分析", value=True)
        
        # SOP 參數：支援多模式選擇
        sop_params = {'modes': []}
        if analyze_sop:
            st.caption("選擇要分析的 SOP 模式（可複選）")
            
            # IDLE 模式
            sop_idle = st.checkbox("🔇 IDLE 模式", value=False, key="sop_idle")
            if sop_idle:
                sop_params['modes'].append('IDLE')
                sop_params['idle_spec'] = st.number_input(
                    "IDLE SPEC 管制線 (dBA)", value=22.0, step=0.5, key="idle_spec"
                )
            
            # UE 模式
            sop_ue = st.checkbox("👤 UE 模式", value=False, key="sop_ue")
            if sop_ue:
                sop_params['modes'].append('UE')
                sop_params['ue_spec'] = st.number_input(
                    "UE SPEC 管制線 (dBA)", value=22.0, step=0.5, key="ue_spec"
                )
            
            # Workload 模式
            sop_workload = st.checkbox("⚡ Workload 模式", value=True, key="sop_workload")
            if sop_workload:
                sop_params['modes'].append('Workload')
                sop_params['work_spec_fail'] = st.number_input(
                    "Fail Rate SPEC (dBA)", value=22.0, step=0.5, key="work_spec_fail"
                )
                sop_params['work_spec_max'] = st.number_input(
                    "Max Leq SPEC (dBA)", value=28.0, step=0.5, key="work_spec_max"
                )
            
            # 向後兼容：設定 mode 為第一個選擇的模式
            if sop_params['modes']:
                sop_params['mode'] = sop_params['modes'][0]
        
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
        st.subheader("📄 報告生成")
        
        if st.session_state.get('audio_loaded', False):
            # Excel 報告
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Excel 報告", key="btn_gen_excel", use_container_width=True):
                    with st.spinner("正在生成 Excel 報告..."):
                        # 獲取校準偏移
                        excel_cal_offset = st.session_state.get('calibration_offset', 0.0)
                        
                        # 構建完整的分析設定
                        analysis_settings = {
                            'use_a_weighting': use_a_weighting,
                            'spectrum_mode': spectrum_mode,
                            'window_function': window_function,
                            'n_fft': n_fft,
                            'ecma_standard': ecma_standard,
                            'spectrogram_spl_offset': spectrogram_spl_offset,
                            'highpass_cutoff': highpass_cutoff
                        }
                        
                        report_data, error = generate_excel_report(
                            st.session_state.audio_original,
                            st.session_state.sr,
                            filename=st.session_state.get('audio_filename', "audio.wav"),
                            sop_params=sop_params,
                            calibration_offset=excel_cal_offset,
                            analysis_settings=analysis_settings
                        )
                        
                        if error:
                            st.error(error)
                        else:
                            st.session_state['report_xlsx'] = report_data
                            st.success("✅ Excel 報告已生成")

                if 'report_xlsx' in st.session_state:
                    st.download_button(
                        label="⬇️ 下載 Excel",
                        data=st.session_state['report_xlsx'],
                        file_name=f"Report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
            
            with col2:
                if st.button("📑 PDF 報告", key="btn_gen_pdf", use_container_width=True):
                    with st.spinner("正在生成 PDF 報告（含圖表）..."):
                        # 強制重新載入 pdf_report 模組，確保使用最新代碼
                        import importlib
                        import utils.pdf_report as pdf_report_module
                        importlib.reload(pdf_report_module)
                        
                        pdf_data, error = pdf_report_module.generate_pdf_report(
                            st.session_state.audio_original,
                            st.session_state.sr,
                            filename=st.session_state.get('audio_filename', "audio.wav"),
                            sop_params=sop_params if analyze_sop else None,
                            analyze_discrete_tone_flag=analyze_discrete_tone,
                            calibration_offset=calibration_offset,
                            leq_settings=leq_settings,
                            use_a_weighting=use_a_weighting,
                            spectrum_mode=spectrum_mode,
                            window_function=window_function,
                            n_fft=n_fft,
                            fft_chart=st.session_state.get('fft_chart_figure', None),
                            level_time_chart=st.session_state.get('level_time_chart_figure', None),
                            octave_chart=st.session_state.get('octave_chart_figure', None),
                            ecma_standard=ecma_standard
                        )
                        
                        if error:
                            st.error(f"PDF 生成失敗: {error}")
                        else:
                            st.session_state['report_pdf'] = pdf_data
                            st.success("✅ PDF 報告已生成")

                if 'report_pdf' in st.session_state:
                    st.download_button(
                        label="⬇️ 下載 PDF",
                        data=st.session_state['report_pdf'],
                        file_name=f"Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
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
                # 使用 streamlit components 執行 JavaScript 滾動到頂部
                import streamlit.components.v1 as components
                components.html("""
                    <script>
                        // 滾動到頁面頂部
                        window.parent.document.querySelector('section.main').scrollTo({
                            top: 0,
                            behavior: 'smooth'
                        });
                    </script>
                """, height=0)
                
                # Streamlit 原生的 spinner 會在右上角顯示 "Running..."
                with st.spinner("🔄 分析中，請稍候..."):
                    render_analysis_results(
                        highpass_cutoff,
                        analyze_noise,
                        analyze_spectrum,
                        analyze_discrete_tone,
                        analyze_sop,
                        sop_params,
                        analyze_band_filter,
                        removed_bands,
                        use_a_weighting,
                        spectrum_mode,
                        window_function,
                        n_fft,
                        ecma_standard,
                        spectrogram_z_range,
                        spectrogram_spl_offset,
                        leq_settings,
                        smooth_window_size
                    )
        else:
            # 批次模式
            st.success(f"✅ 已上傳 **{len(uploaded_files)}** 個檔案，準備進行批次分析")
            if st.button(f"🚀 開始批次分析", type="primary", use_container_width=True):
                process_batch_analysis(uploaded_files, analyze_sop, sop_params)
            
            if st.session_state.get('batch_data'):
                render_batch_dashboard(
                    highpass_cutoff,
                    analyze_noise,
                    analyze_spectrum,
                    analyze_discrete_tone,
                    analyze_sop,
                    sop_params,
                    False,  # analyze_high_freq (deprecated)
                    analyze_band_filter,
                    removed_bands,
                    use_a_weighting,
                    spectrum_mode,
                    window_function,
                    n_fft,
                    smooth_window_size
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


def get_band_frequencies(band_keys):
    """將頻帶 Key 轉換為實際的 1/3 Octave 中心頻率列表"""
    frequencies = []
    mapping = {
        'low_freq': [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400],
        'mid_freq': [500, 630, 800, 1000, 1250, 1600, 2000],
        'mid_high_freq': [2500, 3150, 4000, 5000],
        'high_freq': [6300, 8000, 10000, 12500],
        'ultra_high_freq': [16000, 20000]
    }
    for key in band_keys:
        frequencies.extend(mapping.get(key, []))
    return frequencies


def render_analysis_results(highpass_cutoff, analyze_noise, analyze_spectrum, 
                            analyze_discrete_tone, analyze_sop, sop_params,
                            analyze_band_filter, removed_bands, use_a_weighting=True,
                            spectrum_mode='average', window_function='hann', n_fft=8192,
                            ecma_standard='ECMA-74', spectrogram_z_range=None,
                            spectrogram_spl_offset=0.0, leq_settings=None,
                            smooth_window_size=1):
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
        window_function: 窗函數類型
        n_fft: FFT 點數
        ecma_standard: ECMA 標準版本
        spectrogram_z_range: Spectrogram 顯示範圍
        spectrogram_spl_offset: Spectrogram SPL 偏移
        leq_settings: Leq 判定設定 (spec/tag)
        smooth_window_size: Level vs Time 平滑視窗大小
    """
    # 確保 session_state 中有音檔資料
    if not st.session_state.audio_loaded:
        st.warning("⚠️ 請先上傳並載入音訊檔案。")
        return

    # 從 session_state 獲取原始音訊
    audio_original = st.session_state.audio_original
    sr = st.session_state.sr
    validation = st.session_state.validation
    
    # === 音訊前處理 (過濾) ===
    audio_processed = audio_original  # 初始為原始音訊 (不要修改原始資料)
    
    # 1. 應用帶阻濾波器 (頻帶過濾模擬)
    if analyze_band_filter and removed_bands:
        from core.band_analyzer import apply_band_rejection
        target_frequencies = get_band_frequencies(removed_bands)
        audio_processed = apply_band_rejection(audio_processed, sr, target_frequencies)
        st.warning(f"⚠️ **頻帶過濾啟用**: 已移除 {len(target_frequencies)} 個 1/3 倍頻程頻帶 (選擇了 {len(removed_bands)} 個區域)。分析結果基於過濾後的音訊。")
    
    # 2. 應用高通濾波 (全域設定，預設通常建議 20Hz)
    # 邏輯修正：當「頻帶過濾模擬」啟用時，忽略此設定，避免雙重濾波造成混淆
    is_band_filter_active = analyze_band_filter and removed_bands
    
    if highpass_cutoff > 0 and not is_band_filter_active:
        from scipy.signal import butter, sosfiltfilt
        sos = butter(4, highpass_cutoff, 'hp', fs=sr, output='sos')
        audio_processed = sosfiltfilt(sos, audio_processed)
        
        # 當頻率較高時顯示資訊
        if highpass_cutoff > 20: 
            st.info(f"🔊 **高通濾波已啟用**: 截止頻率 {highpass_cutoff} Hz")
    elif highpass_cutoff > 20 and is_band_filter_active:
        st.caption(f"ℹ️ 高通濾波設定 ({highpass_cutoff} Hz) 已暫時忽略，因為頻帶過濾模擬正在運作中。")

    
    # 顯示加權模式
    pass
    
    # 顯示音檔資訊
    display_audio_info(validation)
    
    # === 同步音訊播放器 (帶 Spectrogram 進度線) ===
    from ui.audio_player import create_audio_player_with_spectrogram, create_simple_audio_player
    
    # 獲取校準偏移
    player_cal_offset = st.session_state.get('calibration_offset', 0.0)
    
    # 獲取 SPL 偏移（如果啟用 dB SPL 模式）
    player_spl_offset = spectrogram_spl_offset
    
    if analyze_band_filter and removed_bands:
        # 有頻帶過濾時顯示兩個播放器
        col1, col2 = st.columns(2)
        with col1:
            st.caption("🎧 **過濾後音訊** (基於此進行分析)")
            create_audio_player_with_spectrogram(audio_processed, sr, "🎵 過濾後音訊播放器", 
                                                  calibration_offset=player_cal_offset,
                                                  use_a_weighting=use_a_weighting,
                                                  spl_offset=player_spl_offset)
        with col2:
            st.caption("🔊 **原始音訊** (對照參考)")
            create_audio_player_with_spectrogram(audio_original, sr, "🔊 原始音訊播放器", 
                                                  calibration_offset=player_cal_offset,
                                                  use_a_weighting=use_a_weighting,
                                                  spl_offset=player_spl_offset)
    else:
        # 只顯示一個播放器
        create_audio_player_with_spectrogram(audio_processed, sr, "🎵 音訊播放器 (點擊頻譜圖可跳轉)", 
                                              calibration_offset=player_cal_offset,
                                              use_a_weighting=use_a_weighting,
                                              spl_offset=player_spl_offset)
    
    st.markdown("---")
    
    # 執行各項分析 (使用過濾後的音訊，傳入 A-weighting 設定)
    if analyze_noise:
        run_noise_analysis(audio_processed, sr, use_a_weighting, leq_settings)

    # 產生唯一的 Key Suffix，確保過濾器參數變更時圖表會強制重繪
    filter_key_suffix = f"{str(removed_bands)}" if analyze_band_filter and removed_bands else "raw"

    if analyze_spectrum:
        run_spectrum_analysis(
            audio_processed, sr, use_a_weighting, spectrum_mode, window_function, n_fft, 
            spectrogram_z_range, spectrogram_spl_offset=spectrogram_spl_offset, 
            smooth_window=smooth_window_size,
            highpass_cutoff=highpass_cutoff,
            calibration_offset=st.session_state.get('calibration_offset', 0.0),
            key_suffix=filter_key_suffix
        )

    if analyze_discrete_tone:
        run_discrete_tone_analysis(
            audio_processed, sr, spectrum_mode, window_function, n_fft, use_a_weighting, 
            ecma_standard, highpass_cutoff=highpass_cutoff,
            key_suffix=filter_key_suffix,
            removed_bands_keys=removed_bands if analyze_band_filter else None
        )

    if analyze_sop:
        run_sop_analysis(audio_processed, sr, sop_params)

    # 如果有頻帶過濾，顯示原始 vs 過濾後對比
    if analyze_band_filter and removed_bands:
        run_band_filter_comparison(audio_original, audio_processed, sr, removed_bands)

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


def run_noise_analysis(audio, sr, use_a_weighting=True, leq_settings=None):
    """執行噪音等級分析"""
    from core.noise_level import calculate_noise_level
    
    result = calculate_noise_level(audio, sr, apply_weighting=use_a_weighting)
    
    # 取得校準偏移
    cal_offset = st.session_state.get('calibration_offset', 0.0)
    
    # 套用校準偏移
    leq = result['leq_dba'] + cal_offset
    lmax = result['lmax_dba'] + cal_offset
    lmin = result['lmin_dba'] + cal_offset
    l10 = result['l10'] + cal_offset
    l90 = result['l90'] + cal_offset
    
    # 動態單位標籤
    unit_label = "dB(A)" if use_a_weighting else "dB"
    
    st.subheader(f"🔊 噪音等級分析 {unit_label}")
    
    # 顯示校準狀態
    if cal_offset != 0:
        st.caption(f"📌 已套用校準偏移: **{cal_offset:+.1f} dB**")
    
    # 定義資訊卡片函數
    def card(label, value, unit, description, is_primary=False, is_fail=False):
        if is_fail:
            border_color = "#E74C3C"  # 紅色邊框
            bg_color = "#FDEDEC"      # 淺紅背景
            text_color = "#C0392B"    # 深紅文字
            box_shadow = "0 4px 6px rgba(231, 76, 60, 0.2)"
        elif is_primary:
            border_color = "#4A90E2"
            bg_color = "#F0F7FF"
            text_color = "#2c3e50"
            box_shadow = "0 4px 6px rgba(0,0,0,0.1)"
        else:
            border_color = "#E0E0E0"
            bg_color = "#FFFFFF"
            text_color = "#2c3e50"
            box_shadow = "0 1px 3px rgba(0,0,0,0.05)"
        
        return f"""
        <div style="
            border: 2px solid {border_color};
            border-radius: 10px;
            padding: 15px 10px;
            text-align: center;
            background-color: {bg_color};
            box-shadow: {box_shadow};
            height: 100%;
            display: flex;
            flex-direction: column;
            justify_content: center;
            align_items: center;
        ">
            <div style="font-size: 14px; color: #888; font-weight: bold; margin-bottom: 5px;">{label}</div>
            <div style="font-size: 28px; font-weight: bold; color: {text_color}; margin: 5px 0;">{value}</div>
            <div style="font-size: 12px; color: #666; background: rgba(0,0,0,0.05); padding: 2px 8px; border-radius: 10px; display: inline-block;">{unit}</div>
            <div style="font-size: 11px; color: #999; margin-top: 8px;">{description}</div>
        </div>
        """

    # 判定 Leq 是否超標
    leq_fail = False
    
    # 解析設定
    leq_spec = None
    leq_tag = ""
    if leq_settings and leq_settings.get('spec'):
        leq_spec = leq_settings.get('spec')
        leq_tag = leq_settings.get('tag', '')

    leq_desc_line1 = "等效連續音壓 (平均)"
    if leq_tag:
        leq_desc_line1 = f"<b>{leq_tag}</b>"
        
    leq_desc = leq_desc_line1
    if leq_spec is not None:
        if leq > leq_spec:
            leq_fail = True
            leq_desc += f"<br>⚠️ 超標 ({leq_spec} dB)"
        else:
            leq_desc += f"<br>✅ 合格 ({leq_spec} dB)"

    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(card("Leq", f"{leq:.1f}", unit_label, leq_desc, is_primary=True, is_fail=leq_fail), unsafe_allow_html=True)
    with col2:
        st.markdown(card("Lmax", f"{lmax:.1f}", unit_label, "最大音壓級"), unsafe_allow_html=True)
    with col3:
        st.markdown(card("Lmin", f"{lmin:.1f}", unit_label, "背景噪音參考"), unsafe_allow_html=True)
    with col4:
        st.markdown(card("L10", f"{l10:.1f}", unit_label, "峰值噪音 (10%)"), unsafe_allow_html=True)
    with col5:
        st.markdown(card("L90", f"{l90:.1f}", unit_label, "背景持續 (90%)"), unsafe_allow_html=True)
    
    st.markdown("---")


def run_spectrum_analysis(audio, sr, use_a_weighting=True, 
                          spectrum_mode='average', window_function='hann', n_fft=8192,
                          spectrogram_z_range=None, calibration_offset=0.0,
                          spectrogram_spl_offset=0.0, smooth_window=1,
                          highpass_cutoff=0, key_suffix=""):
    """執行頻譜分析 - 多種圖表即時切換
    
    Args:
        audio: 音訊資料
        sr: 取樣率
        use_a_weighting: 是否套用 A-weighting 加權
        spectrum_mode: 分析模式 (average/peak_hold/psd)
        window_function: 窗函數 (hann/hamming/blackman/flattop)
        n_fft: FFT 點數 (決定頻率解析度)
        spectrogram_z_range: Spectrogram 色彩範圍 (z_min, z_max)
        calibration_offset: 校準偏移 (dB)
        spectrogram_spl_offset: Spectrogram dB SPL 偏移 (dB)
        smooth_window: Level vs Time 平滑視窗大小
    """
    from core.fft import compute_spectrum_with_mode, apply_a_weighting
    from core.noise_level import calculate_noise_level
    from utils.interactive_plots import (
        create_interactive_spectrum,
        create_waveform_chart,
        create_spectrogram_chart,
        create_a_weighting_chart,
        create_octave_band_chart,
        create_waterfall_3d_chart,
        create_combined_analysis_chart,
        create_level_vs_time_chart,
        create_spectrum_with_leq_line
    )
    import numpy as np
    
    # 分析模式對應的標籤
    mode_labels = {
        'average': 'FFT Average',
        'peak_hold': 'FFT Peak Hold',
        'psd': 'PSD'
    }
    mode_label = mode_labels.get(spectrum_mode, spectrum_mode)
    
    # 計算頻率解析度
    freq_resolution = sr / n_fft
    
    # 使用指定模式、窗函數和 FFT 點數計算頻譜
    frequencies, magnitudes_db, unit = compute_spectrum_with_mode(
        audio, sr, mode=spectrum_mode, n_fft=n_fft, window=window_function
    )
    # 套用 A-weighting (如果啟用)
    if use_a_weighting:
        magnitudes_db = apply_a_weighting(frequencies, magnitudes_db)
        weight_label = f"{unit}(A)" if unit != 'dB/Hz' else "dB(A)/Hz"
    else:
        weight_label = unit
    
    # 套用校準偏移
    magnitudes_db = magnitudes_db + calibration_offset
    cal_offset = calibration_offset
    
    st.subheader(f"📈 頻譜分析 [{mode_label}] - {weight_label}")
    st.caption(f"💡 模式: {mode_label} | 窗函數: {window_function.capitalize()} | 頻率解析度: {freq_resolution:.2f} Hz")
    
    # 準備其他分頁所需的圖表
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        f"📊 {mode_label}", 
        "📈 Level vs Time", 
        "🌊 波形圖", 
        "🔥 Spectrogram", 
        "🎼 1/3 倍頻程 (dB(A))",
        "🌀 3D Waterfall",
        "📑 綜合視圖"
    ])
    
    with tab1:
        # 決定顯示範圍
        x_min = max(20, highpass_cutoff)
        
        # 互動式頻譜圖
        fig = create_interactive_spectrum(
            frequencies, magnitudes_db,
            title=f"頻譜分析 [{mode_label}] - {weight_label} (Res: {freq_resolution:.1f}Hz)",
            ylabel=f"幅度 ({weight_label})",
            freq_range=(x_min, 20000)
        )
        st.plotly_chart(fig, use_container_width=True, key=f"spectrum_main_{highpass_cutoff}_{key_suffix}")
        
        # 保存 FFT 圖表到 session_state，供 PDF 報告使用
        st.session_state['fft_chart_figure'] = fig
        
        st.caption(f"ℹ️ 設定參數: Window={window_function}, N_FFT={n_fft}, Mode={spectrum_mode}")

    with tab2:
        # Level vs Time 圖表（使用自定義平滑參數）
        # 使用 audio_processed 以反映濾波效果
        level_time_fig = create_level_vs_time_chart(audio, sr, smooth_window=smooth_window, calibration_offset=cal_offset, use_a_weighting=use_a_weighting)
        st.plotly_chart(level_time_fig, use_container_width=True, key="level_vs_time")
        
        # 保存 Level vs Time 圖表到 session_state，供 PDF 報告使用
        st.session_state['level_time_chart_figure'] = level_time_fig
    
    with tab3:
        # 波形圖
        wave_fig = create_waveform_chart(audio, sr, title="音訊波形圖 (Waveform)")
        st.plotly_chart(wave_fig, use_container_width=True, key="waveform")
        
    with tab4:
        # Spectrogram
        spec_fig = create_spectrogram_chart(
            audio, sr, 
            use_a_weighting=use_a_weighting,
            z_range=spectrogram_z_range,
            calibration_offset=cal_offset,
            spl_offset=spectrogram_spl_offset
        )
        st.plotly_chart(spec_fig, use_container_width=True, key="spectrogram_main")
        
    with tab5:
        # 1/3 Octave
        octave_fig = create_octave_band_chart(
            audio, sr, 
            use_a_weighting=True, # 倍頻程通常使用 A-weighting
            calibration_offset=cal_offset
        )
        st.plotly_chart(octave_fig, use_container_width=True, key="octave_main")
        
        # 保存 1/3 Octave 圖表到 session_state，供 PDF 報告使用
        st.session_state['octave_chart_figure'] = octave_fig
        
    with tab6:
        # 3D Waterfall
        waterfall_fig = create_waterfall_3d_chart(audio, sr)
        st.plotly_chart(waterfall_fig, use_container_width=True, key="waterfall")
        st.info("💡 3D Waterfall 圖可旋轉、縮放。拖曳可改變視角，滾輪縮放。")
    
    with tab7:
        combined_fig = create_combined_analysis_chart(
            audio, sr, frequencies, magnitudes_db,
            calibration_offset=cal_offset,
            spl_offset=spectrogram_spl_offset,
            z_range=spectrogram_z_range,
            use_a_weighting=use_a_weighting,
            smooth_window=smooth_window
        )
        st.plotly_chart(combined_fig, use_container_width=True, key="combined")
    
    st.markdown("---")


def run_discrete_tone_analysis(audio, sr, spectrum_mode='average', window_function='hann', n_fft=8192, use_a_weighting=True, ecma_standard='ECMA-74', highpass_cutoff=0, key_suffix="", removed_bands_keys=None):
    """執行 Discrete Tone 檢測
    
    Args:
        audio: 音訊資料
        sr: 取樣率
        spectrum_mode: 頻譜分析模式 (average/peak_hold/psd)
        window_function: 窗函數 (hann/hamming/blackman/flattop)
        n_fft: FFT 點數
        use_a_weighting: 是否使用 A-weighting
        ecma_standard: 使用的判定標準 (ECMA-74 或 ECMA-418)
        highpass_cutoff: 高通濾波截止頻率 (用於圖表顯示)
    """
    from core.discrete_tone import detect_discrete_tones
    from core.fft import compute_average_spectrum, compute_peak_hold_spectrum, compute_psd, get_frequency_range
    from utils.interactive_plots import create_discrete_tone_chart
    
    # 使用當前選擇的頻譜模式和 ECMA 標準進行 Discrete Tone 檢測
    result = detect_discrete_tones(audio, sr, spectrum_mode=spectrum_mode, 
                                   window_function=window_function, n_fft=n_fft,
                                   ecma_standard=ecma_standard)
    
    # 模式名稱對應
    mode_names = {
        'average': 'FFT Average',
        'peak_hold': 'FFT Peak Hold',
        'psd': 'PSD'
    }
    mode_display = mode_names.get(spectrum_mode, spectrum_mode)
    
    # 單位標籤
    unit_label = "dB(A)" if use_a_weighting else "dB"
    
    # 分析模式對應的標籤
    spectrum_mode_labels = {
        'average': 'FFT Average',
        'peak_hold': 'FFT Peak Hold',
        'psd': 'PSD'
    }

    # 執行檢測
    result = detect_discrete_tones(
        audio, sr, 
        ecma_standard=ecma_standard,
        n_fft=n_fft,
        spectrum_mode=spectrum_mode,
        window_function=window_function,
        use_a_weighting=use_a_weighting # 已支援：僅影響回傳的 spectrum 數據用於統一顯示
    )
    
    # === 過濾掉位於「已移除頻帶」內的假警報 ===
    if result["tone_detected"] and removed_bands_keys:
        # 定義頻帶範圍 (必須與 Sidebar 定義一致)
        # 為了安全起見，稍微放寬邊界，確保邊緣的鋸齒峰值也被過濾
        BAND_RANGES = {
            'low_freq': (0, 500),          # 20-500Hz (延伸至0以防萬一)
            'mid_freq': (500, 2000),       # 500-2kHz
            'mid_high_freq': (2000, 6000), # 2-6kHz
            'high_freq': (6000, 12000),    # 6-12kHz
            'ultra_high_freq': (12000, 24000) # 12-20kHz+
        }
        
        filtered_tones = []
        for tone in result['tones']:
            tone_freq = tone['frequency']
            is_removed = False
            
            # 檢查此 Tone 是否落入任何一個被移除的區間
            for band_key in removed_bands_keys:
                if band_key in BAND_RANGES:
                    f_min, f_max = BAND_RANGES[band_key]
                    if f_min <= tone_freq <= f_max:
                        is_removed = True
                        break
            
            # 只有未被移除的 Tone 才保留
            if not is_removed:
                filtered_tones.append(tone)
                
        # 更新結果
        result['tones'] = filtered_tones
        result['tone_detected'] = len(filtered_tones) > 0

    # 取得校準偏移
    discrete_tone_cal_offset = st.session_state.get('calibration_offset', 0.0)
    
    # 顯示檢測結果
    st.subheader(f"🎵 Discrete Tone 檢測 ({ecma_standard}) - {spectrum_mode_labels.get(spectrum_mode, spectrum_mode)}")
    st.caption(f"💡 提示: 紅色星號標記超過門檻的 Discrete Tone，灰色三角形為候選峰值")
    
    # 決定顯示範圍 X 軸
    x_min = max(20, highpass_cutoff)
    
    tone_fig = create_discrete_tone_chart(
        result['frequencies'], 
        result['magnitudes'], 
        result['tones'],
        result.get('all_candidates', []),
        title=f"Discrete Tone 檢測結果 ({ecma_standard} 標準) - {spectrum_mode_labels.get(spectrum_mode, spectrum_mode)}",
        use_a_weighting=use_a_weighting,
        ecma_standard=ecma_standard,
        calibration_offset=discrete_tone_cal_offset,
        freq_range=(x_min, 15000)
    )
    st.plotly_chart(tone_fig, use_container_width=True, key=f"discrete_tone_{highpass_cutoff}_{key_suffix}")
    
    # 狀態顯示
    if result["tone_detected"]:
        st.warning(f"⚠️ 偵測到 {len(result['tones'])} 個 Discrete Tone!")
        
        # 顯示詳細列表
        for i, tone in enumerate(result['tones'], 1):
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric(f"Tone #{i} 頻率", f"{tone['frequency']:.0f} Hz")
            with col2:
                st.metric("PR (ΔLp)", f"{tone['prominence']:.1f} dB")
            with col3:
                st.metric("TNR (ΔLt)", f"{tone.get('tnr', 0):.1f} dB")
            with col4:
                st.metric("判定方法", tone.get('method', 'PR'))
            with col5:
                st.metric("頻帶", tone['band'])
    else:
        st.success("✅ 未偵測到顯著的 Discrete Tone")

    
    # 顯示判定標準
    st.caption(f"📐 判定標準: {result.get('criteria', 'ECMA-418-1')}")
    
    # 顯示候選 Tone
    if result.get("all_candidates"):
        with st.expander("🔍 查看所有候選峰值 (ECMA-418-1 雙準則)"):
            import pandas as pd
            candidates = result["all_candidates"]
            # 處理新舊資料格式
            if candidates and 'tnr' in candidates[0]:
                df = pd.DataFrame(candidates)
                df = df[['frequency', 'prominence', 'tnr', 'pr_threshold', 'tnr_threshold', 'method', 'exceeds_threshold', 'band']]
                df.columns = ["頻率 (Hz)", "PR (dB)", "TNR (dB)", "PR門檻", "TNR門檻", "判定方法", "超過門檻", "頻帶"]
            else:
                df = pd.DataFrame(candidates)
                if not df.empty:
                    df.columns = ["頻率 (Hz)", "突出量 (dB)", "幅度 (dB)", "頻帶", "門檻 (dB)", "超過門檻"]
            st.dataframe(df, use_container_width=True)
    
    st.markdown("---")


def run_sop_analysis(audio, sr, sop_params):
    """執行 ASUS SOP 高頻音分析（支援多模式）"""
    from core.sop_analyzer import analyze_idle_mode, analyze_ue_mode, analyze_workload_mode
    import plotly.graph_objects as go
    import numpy as np
    
    # 取得校準偏移
    cal_offset = st.session_state.get('calibration_offset', 0.0)
    
    # 取得要分析的模式列表
    modes = sop_params.get('modes', [sop_params.get('mode', 'IDLE')])
    
    if not modes:
        st.warning("請選擇至少一個 SOP 模式進行分析")
        return
    
    st.subheader(f"⚡ ASUS SOW 高頻音分析")
    
    # 顯示校準狀態
    if cal_offset != 0:
        st.caption(f"📌 已套用校準偏移: **{cal_offset:+.1f} dB**")
    
    # 儲存分析結果供報告使用
    sop_results = {}
    
    # ===== IDLE 模式 =====
    if 'IDLE' in modes:
        st.markdown("### 🔇 IDLE Mode")
        spec_limit = sop_params.get('idle_spec', 20.0)
        
        adjusted_spec = spec_limit - cal_offset
        result = analyze_idle_mode(audio, sr, adjusted_spec)
        sop_results['IDLE'] = result
        
        max_leq = result['max_leq'] + cal_offset
        leqs_calibrated = np.array(result['leqs']) + cal_offset
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Max Leq", f"{max_leq:.1f} dBA")
        with col2:
            st.metric("判定結果", "PASS" if result['is_pass'] else "FAIL")
        
        if not result['is_pass']:
            st.error(f"❌ 檢測失敗：有部分數據點超過管制線 {spec_limit} dBA")
        else:
            st.success(f"✅ 檢測通過：所有數據點都在管制線 {spec_limit} dBA 以下")
            
        # 繪製趨勢圖
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result['times'], 
            y=leqs_calibrated.tolist(), 
            name="Moving Leq (10s)",
            hovertemplate='<b>%{x:.1f}s</b> | %{y:.1f} dBA<extra></extra>'
        ))
        fig.add_hline(y=spec_limit, line_dash="dash", line_color="red", 
                      annotation_text=f"SPEC: {spec_limit} dBA")
        fig.update_layout(
            title="IDLE Mode 10s Moving Average 趨勢圖", 
            xaxis_title="時間 (秒)", 
            yaxis_title="音壓級 (dBA)",
            xaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor',
                       spikecolor='red', spikethickness=1, spikedash='dot'),
            yaxis=dict(showspikes=False),
            hovermode='x',
            hoverlabel=dict(bgcolor='rgba(255,255,255,0.95)', 
                           bordercolor='rgba(100,100,100,0.3)', font_size=11)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
    
    # ===== UE 模式 =====
    if 'UE' in modes:
        st.markdown("### 👤 UE Mode")
        ue_spec = sop_params.get('ue_spec', 22.0)
        result = analyze_ue_mode(audio, sr)
        sop_results['UE'] = result
        
        leq_calibrated = result['leq'] + cal_offset
        is_pass = leq_calibrated <= ue_spec
        result['is_pass'] = is_pass  # 添加判定結果
        result['spec'] = ue_spec  # 記錄 SPEC
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("整段平均 Leq", f"{leq_calibrated:.1f} dBA")
        with col2:
            st.metric("判定結果", "PASS" if is_pass else "FAIL")
        
        if is_pass:
            st.success(f"✅ 檢測通過：平均 Leq {leq_calibrated:.1f} dBA ≤ 管制線 {ue_spec} dBA")
        else:
            st.error(f"❌ 檢測失敗：平均 Leq {leq_calibrated:.1f} dBA > 管制線 {ue_spec} dBA")
        
        st.info(f"錄音時長: {result['duration']:.1f} 秒")
        st.markdown("---")
    
    # ===== Workload 模式 =====
    if 'Workload' in modes:
        st.markdown("### ⚡ Workload Mode")
        spec_fail = sop_params.get('work_spec_fail', 22.0)
        spec_max = sop_params.get('work_spec_max', 28.0)
        
        adjusted_spec_fail = spec_fail - cal_offset
        adjusted_spec_max = spec_max - cal_offset
        result = analyze_workload_mode(audio, sr, adjusted_spec_fail, adjusted_spec_max)
        sop_results['Workload'] = result
        
        max_leq = result['max_leq'] + cal_offset
        leqs_calibrated = np.array(result['leqs']) + cal_offset
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Max Leq", f"{max_leq:.1f} dBA")
        with col2:
            st.metric(f"Fail Rate (>{spec_fail})", f"{result['fail_rate']}%")
        with col3:
            st.metric("整體結果", "PASS" if result['is_pass'] else "FAIL")
            
        if not result['criteria_max_pass']:
            st.error(f"❌ Max 值超過管制線 {spec_max} dBA")
        if not result['criteria_rate_pass']:
            st.error(f"❌ Fail Rate ({result['fail_rate']}%) 超過 2% 門檻")
            
        # 繪製趨勢圖
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result['times'], 
            y=leqs_calibrated.tolist(), 
            name="Moving Leq (1s)",
            hovertemplate='<b>%{x:.1f}s</b> | %{y:.1f} dBA<extra></extra>'
        ))
        fig.add_hline(y=spec_fail, line_dash="dot", line_color="orange", 
                      annotation_text=f"Fail Rate Limit: {spec_fail} dBA")
        fig.add_hline(y=spec_max, line_dash="dash", line_color="red", 
                      annotation_text=f"Max Limit: {spec_max} dBA")
        fig.update_layout(
            title="Workload Mode 1s Moving Average 趨勢圖", 
            xaxis_title="時間 (秒)", 
            yaxis_title="音壓級 (dBA)",
            xaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor',
                       spikecolor='red', spikethickness=1, spikedash='dot'),
            yaxis=dict(showspikes=False),
            hovermode='x',
            hoverlabel=dict(bgcolor='rgba(255,255,255,0.95)', 
                           bordercolor='rgba(100,100,100,0.3)', font_size=11)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
    
    # 儲存結果到 session_state 供報告使用
    st.session_state['sop_results'] = sop_results



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


def process_batch_analysis(uploaded_files, analyze_sop=False, sop_params=None):
    """執行批次分析"""
    import pandas as pd
    import tempfile
    import os
    from core.audio_loader import load_audio, validate_audio
    from core.noise_level import calculate_noise_level
    from core.fft import compute_average_spectrum
    from core.sop_analyzer import analyze_idle_mode, analyze_ue_mode, analyze_workload_mode
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
            
            # 4. SOP Analysis
            sop_res = None
            if analyze_sop:
                mode = sop_params.get('mode', 'IDLE')
                if mode == "IDLE":
                    sop_res = analyze_idle_mode(audio, sr, sop_params.get('idle_spec', 20.0))
                elif mode == "UE":
                    sop_res = analyze_ue_mode(audio, sr)
                elif mode == "Workload":
                    sop_res = analyze_workload_mode(audio, sr, sop_params.get('work_spec_fail', 22.0), sop_params.get('work_spec_max', 28.0))
            
            # 5. Spectrum
            freqs, mags = compute_average_spectrum(audio, sr)
            
            # 6. 1/3 Octave Bands
            octave = compute_octave_bands(audio, sr, use_a_weighting=True)
            
            # Store Result
            batch_results[file.name] = {
                "noise": noise,
                "sop": sop_res,
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
    analyze_sop,
    sop_params,
    analyze_high_freq,
    analyze_band_filter,
    removed_bands,
    use_a_weighting,
    spectrum_mode,
    window_function,
    n_fft=8192,
    smooth_window_size=1
):
    """顯示批次分析儀表板"""
    import plotly.graph_objects as go
    import numpy as np
    
    data = st.session_state.get('batch_data', {})
    if not data:
        return

    st.header("📊 批次分析比較儀表板")
    
    # 1. Comparison Table
    st.subheader("1. 數據總表")
    table_rows = []
    
    for name, res in data.items():
        n = res['noise']
        sop = res.get('sop')
        
        row = {
            "Filename": name,
            "Leq (dBA)": n['leq_dba'],
            "Lmax": n['lmax_dba'],
            "L90": n['l90']
        }
        
        # SOP Result
        if sop:
            row["SOP Mode"] = sop['mode']
            row["SOP Result"] = "PASS" if sop.get('is_pass', True) else "FAIL"
            if sop['mode'] == "UE":
                row["SOP Val (Avg)"] = sop['leq']
            else:
                row["SOP Val (Max)"] = sop['max_leq']
        
        table_rows.append(row)
    
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
            
            # 套用平滑處理 (比較模式同樣受全域設定影響)
            if smooth_window_size > 1:
                kernel = np.ones(smooth_window_size) / smooth_window_size
                levels = np.convolve(levels, kernel, mode='same')
            
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
                fig_spec = create_spectrogram_chart(audio_data, sr_data, title=f"Spectrogram: {name}", use_a_weighting=use_a_weighting)
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
            analyze_sop,
            sop_params,
            analyze_band_filter,
            removed_bands,
            use_a_weighting,
            spectrum_mode,
            window_function,
            n_fft
        )



if __name__ == "__main__":
    main()


