# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - 互動式視覺化模組

使用 Plotly 實現互動式圖表：
- 滑鼠十字座標追蹤
- 縮放、平移
- 數值顯示
"""

from typing import Tuple, List, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.band_analyzer import compute_octave_bands


def create_interactive_spectrum(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    title: str = "頻譜圖",
    log_scale: bool = True,
    freq_range: Tuple[float, float] = None,
    show_grid: bool = True,
    ylabel: str = "幅度 (dB)"
) -> go.Figure:
    """建立互動式頻譜圖

    Args:
        frequencies: 頻率陣列 (Hz)
        magnitudes: 能量陣列 (dB)
        title: 圖表標題
        log_scale: 是否使用對數頻率軸
        freq_range: 顯示的頻率範圍
        show_grid: 是否顯示網格
        ylabel: Y 軸標籤

    Returns:
        go.Figure: Plotly 圖表物件
    """
    fig = go.Figure()

    # 新增頻譜線
    fig.add_trace(go.Scatter(
        x=frequencies,
        y=magnitudes,
        mode='lines',
        name='頻譜',
        line=dict(color='#1f77b4', width=1),
        hovertemplate='頻率: %{x:.1f} Hz<br>幅度: %{y:.1f} dB<extra></extra>'
    ))

    # 設定頻率範圍
    if freq_range is None:
        freq_range = (max(20, frequencies[0]), min(20000, frequencies[-1]))

    # 計算 Y 軸範圍（確保完整顯示，不會截斷峰值）
    valid_mags = magnitudes[np.isfinite(magnitudes)]
    if len(valid_mags) > 0:
        # 計算 Y 軸範圍（確保完整顯示，不會截斷峰值）
        # 改進：能夠適應濾波器產生的深谷 (Deep Notches)
        y_max = np.max(valid_mags)
        y_min_real = np.min(valid_mags)
        
        # 如果有極低值（如濾波後），確保顯示出來，但設定合理下限避免 -inf
        # 一般聲學分析底限約 -100dB 到 -120dB
        target_min = max(y_min_real, -120)
        
        # 上方留 10dB (或 20%) 空間
        y_range = y_max - target_min
        y_axis_max = y_max + max(10, y_range * 0.1)
        
        # 下方留 10dB (或 10%) 空間
        y_axis_min = target_min - max(10, y_range * 0.1)
    else:
        y_axis_min = -50
        y_axis_max = 50

    # 設定布局
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        xaxis=dict(
            title='頻率 (Hz)',
            type='log' if log_scale else 'linear',
            range=[np.log10(freq_range[0]), np.log10(freq_range[1])] if log_scale else list(freq_range),
            showgrid=show_grid,
            gridcolor='rgba(128, 128, 128, 0.3)',
            tickformat='.0f',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',  # 垂直線跟著滑鼠
            spikecolor='red',
            spikethickness=1,
            spikedash='dot'
        ),
        yaxis=dict(
            title=ylabel,
            range=[y_axis_min, y_axis_max],
            showgrid=show_grid,
            gridcolor='rgba(128, 128, 128, 0.3)',
            showspikes=False  # 不需要 Y 軸 spike
        ),
        hovermode='x',  # 在任意 Y 位置都可觸發 hover
        dragmode='zoom',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=60, b=60),
        modebar=dict(
            orientation='h',
            bgcolor='rgba(255,255,255,0.7)'
        )
    )
    
    # 設定緊湊的 hover 資訊
    fig.update_traces(
        hovertemplate='<b>%{x:.0f} Hz</b> | %{y:.1f} dB<extra></extra>'  # 緊湊格式：頻率 | dB
    )
    fig.update_layout(
        hoverlabel=dict(
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(100,100,100,0.3)',
            font_size=11,
            font_family='Arial'
        )
    )

    return fig


def create_comparison_spectrum(
    frequencies: np.ndarray,
    magnitudes_original: np.ndarray,
    magnitudes_filtered: np.ndarray,
    cutoff_freq: float,
    title: str = "高通濾波前後對比"
) -> go.Figure:
    """建立互動式濾波前後對比圖

    Args:
        frequencies: 頻率陣列 (Hz)
        magnitudes_original: 原始能量陣列 (dB)
        magnitudes_filtered: 濾波後能量陣列 (dB)
        cutoff_freq: 截止頻率 (Hz)
        title: 圖表標題

    Returns:
        go.Figure: Plotly 圖表物件
    """
    fig = go.Figure()

    # 原始頻譜
    fig.add_trace(go.Scatter(
        x=frequencies,
        y=magnitudes_original,
        mode='lines',
        name='原始頻譜',
        line=dict(color='#1f77b4', width=1.5),
        opacity=0.7,
        hovertemplate='原始<br>頻率: %{x:.1f} Hz<br>幅度: %{y:.1f} dB<extra></extra>'
    ))

    # 濾波後頻譜
    fig.add_trace(go.Scatter(
        x=frequencies,
        y=magnitudes_filtered,
        mode='lines',
        name='濾波後頻譜',
        line=dict(color='#ff7f0e', width=2),
        hovertemplate='濾波後<br>頻率: %{x:.1f} Hz<br>幅度: %{y:.1f} dB<extra></extra>'
    ))

    # 截止頻率垂直線
    fig.add_vline(
        x=cutoff_freq,
        line=dict(color='red', width=2, dash='dash'),
        annotation_text=f'截止頻率: {cutoff_freq} Hz',
        annotation_position='top right',
        annotation_font=dict(color='red', size=12)
    )

    # 填充截止頻率以下區域 (被濾除的部分)
    fig.add_vrect(
        x0=20, x1=cutoff_freq,
        fillcolor='rgba(255, 0, 0, 0.1)',
        line_width=0,
        annotation_text='濾除區域',
        annotation_position='bottom left',
        annotation_font=dict(color='rgba(255,0,0,0.5)', size=10)
    )

    # 設定布局
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        xaxis=dict(
            title='頻率 (Hz)',
            type='log',
            range=[np.log10(20), np.log10(20000)],
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='red',
            spikethickness=1,
            spikedash='dot'
        ),
        yaxis=dict(
            title='幅度 (dB)',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='red',
            spikethickness=1,
            spikedash='dot'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        hovermode='x unified',
        dragmode='zoom',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=80, b=60)
    )

    return fig


def create_spectrogram_interactive(
    times: np.ndarray,
    frequencies: np.ndarray,
    spectrogram_db: np.ndarray,
    title: str = "頻譜瀑布圖 (Spectrogram)",
    fmax: int = 20000
) -> go.Figure:
    """建立互動式 Spectrogram

    Args:
        times: 時間軸 (秒)
        frequencies: 頻率軸 (Hz)
        spectrogram_db: 頻譜圖 (dB)
        title: 圖表標題
        fmax: 最大顯示頻率 (Hz)

    Returns:
        go.Figure: Plotly 圖表物件
    """
    # 限制頻率範圍
    freq_mask = frequencies <= fmax
    frequencies = frequencies[freq_mask]
    spectrogram_db = spectrogram_db[freq_mask, :]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=times,
        y=frequencies,
        z=spectrogram_db,
        colorscale='Viridis',
        # 自適應顏色範圍
        colorbar=dict(
            title=dict(text='幅度 (dB)', side='right')
        ),
        hovertemplate='時間: %{x:.2f}s<br>頻率: %{y:.0f} Hz<br>幅度: %{z:.1f} dB<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        xaxis=dict(
            title='時間 (秒)',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)'
        ),
        yaxis=dict(
            title='頻率 (Hz)',
            type='log',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)'
        ),
        dragmode='zoom',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=100, t=60, b=60)
    )

    return fig


def create_band_energy_chart(
    band_data: dict,
    title: str = "頻帶能量分析"
) -> go.Figure:
    """建立互動式頻帶能量長條圖

    Args:
        band_data: 各頻帶分析結果
        title: 圖表標題

    Returns:
        go.Figure: Plotly 圖表物件
    """
    # 準備資料
    band_display_names = {
        'low_freq': '低頻<br>(20-500Hz)',
        'mid_freq': '中頻<br>(0.5-2kHz)',
        'mid_high_freq': '中高頻<br>(2-6kHz)',
        'high_freq': '高頻<br>(6-12kHz)',
        'ultra_high_freq': '超高頻<br>(12-20kHz)'
    }

    status_colors = {
        'PASS': '#4CAF50',
        'WARNING': '#FF9800',
        'FAIL': '#F44336'
    }

    names = []
    energies = []
    colors = []
    hover_texts = []

    for band_name, data in band_data.items():
        display_name = band_display_names.get(band_name, band_name)
        energy = data.get('energy_db', -100)
        if not np.isfinite(energy):
            energy = -100
        status = data.get('status', 'PASS')
        freq_range = data.get('range', 'N/A')

        names.append(display_name)
        energies.append(energy)
        colors.append(status_colors.get(status, '#9E9E9E'))
        hover_texts.append(
            f"頻帶: {freq_range}<br>"
            f"能量: {energy:.1f} dB<br>"
            f"狀態: {status}"
        )

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=names,
        y=energies,
        marker_color=colors,
        marker_line=dict(color='black', width=1),
        text=[f'{e:.1f} dB' for e in energies],
        textposition='outside',
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_texts
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        xaxis=dict(
            title='頻帶',
            tickangle=0
        ),
        yaxis=dict(
            title='能量 (dB)',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)'
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=60, b=80),
        bargap=0.3
    )

    # 自動調整 Y 軸範圍
    valid_energies = [e for e in energies if np.isfinite(e)]
    if valid_energies:
        y_min = min(valid_energies) - 10
        y_max = max(valid_energies) + 15
        fig.update_yaxes(range=[y_min, y_max])

    return fig


def create_dual_spectrum_comparison(
    freq1: np.ndarray,
    mag1: np.ndarray,
    freq2: np.ndarray,
    mag2: np.ndarray,
    title1: str = "原始頻譜",
    title2: str = "濾波後頻譜"
) -> go.Figure:
    """建立雙頻譜並排對比圖

    Args:
        freq1, mag1: 第一個頻譜數據
        freq2, mag2: 第二個頻譜數據
        title1, title2: 子圖標題

    Returns:
        go.Figure: Plotly 圖表物件
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(title1, title2),
        shared_yaxes=True,
        horizontal_spacing=0.08
    )

    # 原始頻譜
    fig.add_trace(
        go.Scatter(
            x=freq1, y=mag1,
            mode='lines',
            name=title1,
            line=dict(color='#1f77b4', width=1),
            hovertemplate='頻率: %{x:.1f} Hz<br>幅度: %{y:.1f} dB<extra></extra>'
        ),
        row=1, col=1
    )

    # 濾波後頻譜
    fig.add_trace(
        go.Scatter(
            x=freq2, y=mag2,
            mode='lines',
            name=title2,
            line=dict(color='#ff7f0e', width=1),
            hovertemplate='頻率: %{x:.1f} Hz<br>幅度: %{y:.1f} dB<extra></extra>'
        ),
        row=1, col=2
    )

    # 設定兩邊都使用對數軸和十字線
    for col in [1, 2]:
        fig.update_xaxes(
            type='log',
            title='頻率 (Hz)',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            showspikes=True,
            spikemode='across',
            spikecolor='red',
            spikethickness=1,
            spikedash='dot',
            row=1, col=col
        )

    fig.update_yaxes(
        title='幅度 (dB)',
        showgrid=True,
        gridcolor='rgba(128, 128, 128, 0.3)',
        showspikes=True,
        spikemode='across',
        spikecolor='red',
        spikethickness=1,
        spikedash='dot',
        row=1, col=1
    )

    fig.update_layout(
        hovermode='closest',  # 使用 closest 讓十字座標完整顯示
        dragmode='zoom',
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        margin=dict(l=60, r=40, t=60, b=60),
        height=400
    )

    return fig


def create_discrete_tone_chart(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    tones: list,
    all_candidates: list = None,
    title: str = "Discrete Tone 檢測結果",
    use_a_weighting: bool = True,
    ecma_standard: str = 'ECMA-74',
    calibration_offset: float = 0.0,
    freq_range: Tuple[float, float] = None
) -> go.Figure:
    """建立 Discrete Tone 視覺化圖表

    在頻譜圖上標記偵測到的 Discrete Tone 位置

    Args:
        frequencies: 頻率陣列 (Hz)
        magnitudes: 能量陣列 (dB 或 dB(A))
        tones: 超過門檻的 Tone 列表
        all_candidates: 所有候選峰值列表
        title: 圖表標題
        use_a_weighting: 是否使用 A-weighting (影響 Y 軸單位顯示)
        ecma_standard: 使用的判定標準 (ECMA-74 或 ECMA-418)
        calibration_offset: 校準偏移 (dB)

    Returns:
        go.Figure: Plotly 圖表物件
    """
    unit_label = "dB(A)" if use_a_weighting else "dB"
    
    # 套用校準偏移到頻譜
    magnitudes_calibrated = magnitudes + calibration_offset
    
    fig = go.Figure()

    # 繪製頻譜基線
    fig.add_trace(go.Scatter(
        x=frequencies,
        y=magnitudes_calibrated,
        mode='lines',
        name='頻譜',
        line=dict(color='#1f77b4', width=1),
        hovertemplate='頻率: %{x:.1f} Hz<br>幅度: %{y:.1f} dB<extra></extra>'
    ))

    # 標記所有候選峰值 (灰色三角形)
    if all_candidates:
        # 過濾未超過門檻的候選
        non_exceeding = [t for t in all_candidates if not t.get('exceeds_threshold', False)]
        
        if non_exceeding:
            candidate_freqs = [t.get('frequency', 0) for t in non_exceeding]
            # 套用校準偏移到候選峰值
            candidate_mags = [t.get('magnitude_db', 0) + calibration_offset for t in non_exceeding]
            
            # 建立完整的 hover 資訊（顯示校準後數值）
            hover_texts = [
                f"候選峰值<br>"
                f"<b>頻率: {t.get('frequency', 0):.1f} Hz</b><br>"
                f"幅度: {t.get('magnitude_db', 0) + calibration_offset:.1f} dB<br>"
                f"───────────<br>"
                f"PR: {t.get('prominence', 0):.1f} dB (門檻: {t.get('pr_threshold', 0):.1f})<br>"
                f"TNR: {t.get('tnr', 0):.1f} dB (門檻: {t.get('tnr_threshold', 0):.1f})<br>"
                f"頻帶: {t.get('band', 'N/A')}"
                for t in non_exceeding
            ]
            
            fig.add_trace(go.Scatter(
                x=candidate_freqs,
                y=candidate_mags,
                mode='markers',
                name='候選峰值',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='gray',
                    line=dict(color='darkgray', width=1)
                ),
                hovertemplate='%{customdata}<extra></extra>',
                customdata=hover_texts
            ))

    # 標記超過門檻的 Discrete Tone (紅色星形)
    if tones:
        tone_freqs = [t.get('frequency', 0) for t in tones]
        # 套用校準偏移到 Discrete Tone
        tone_mags = [t.get('magnitude_db', 0) + calibration_offset for t in tones]
        tone_proms = [t.get('prominence', 0) for t in tones]
        tone_tnrs = [t.get('tnr', 0) for t in tones]
        tone_pr_thresholds = [t.get('pr_threshold', t.get('threshold', 0)) for t in tones]
        tone_tnr_thresholds = [t.get('tnr_threshold', 0) for t in tones]
        tone_methods = [t.get('method', 'PR') for t in tones]
        tone_bands = [t.get('band', 'N/A') for t in tones]
        
        hover_texts = [
            f"⚠️ Discrete Tone 偵測<br>"
            f"<b>頻率: {f:.1f} Hz</b><br>"
            f"幅度: {m:.1f} dB<br>"
            f"───────────<br>"
            f"PR (ΔLp): {p:.1f} dB (門檻: {pth:.1f})<br>"
            f"TNR (ΔLt): {tnr:.1f} dB (門檻: {tth:.1f})<br>"
            f"判定方法: {method}<br>"
            f"頻帶: {b}"
            for f, m, p, tnr, pth, tth, method, b in zip(
                tone_freqs, tone_mags, tone_proms, tone_tnrs, 
                tone_pr_thresholds, tone_tnr_thresholds, tone_methods, tone_bands
            )
        ]
        
        fig.add_trace(go.Scatter(
            x=tone_freqs,
            y=tone_mags,
            mode='markers+text',
            name='Discrete Tone',
            marker=dict(
                symbol='star',
                size=15,
                color='red',
                line=dict(color='darkred', width=2)
            ),
            text=[f'{f:.0f}Hz' for f in tone_freqs],
            textposition='top center',
            textfont=dict(color='red', size=10),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts
        ))
    # 門檻區域標示 (根據選擇的標準)
    if ecma_standard == 'ECMA-418':
        # ECMA-418-1: PR 門檻由公式計算
        ecma_bands = [
            (89.1, 1000, None, '<1kHz: >9+10×log₁₀(1000/f) dB'),
            (1000, 11220, 9, '≥1kHz: >9 dB')
        ]
        colors = ['rgba(255,100,100,0.08)', 'rgba(255,200,100,0.08)']
    else:
        # ECMA-74: 固定頻帶門檻
        ecma_bands = [
            (89.1, 282, 8, '89-282Hz: >8dB'),
            (282, 893, 5, '282-893Hz: >5dB'),
            (893, 11200, 3, '893-11.2kHz: >3dB')
        ]
        colors = ['rgba(255,0,0,0.05)', 'rgba(255,165,0,0.05)', 'rgba(255,255,0,0.05)']
    
    for i, (low, high, threshold, label) in enumerate(ecma_bands):
        fig.add_vrect(
            x0=low, x1=high,
            fillcolor=colors[i],
            line_width=0,
            layer='below'
        )
        # 在圖例中顯示門檻資訊
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            name=f'PR {label}',
            marker=dict(size=10, color=colors[i].replace('0.05', '0.3').replace('0.08', '0.4')),
            showlegend=True
        ))

    if freq_range is None:
        freq_range = (50, 15000)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        xaxis=dict(
            title='頻率 (Hz)',
            type='log',
            range=[np.log10(max(20, freq_range[0])), np.log10(freq_range[1])],
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',  # 垂直線跟著滑鼠
            spikecolor='red',
            spikethickness=1,
            spikedash='dot'
        ),
        yaxis=dict(
            title=f'幅度 ({unit_label})',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            showspikes=False
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        hovermode='x',  # 在任意 Y 位置都可觸發 hover
        hoverlabel=dict(
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(100,100,100,0.3)',
            font_size=11
        ),
        dragmode='zoom',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    # Discrete Tone 圖表的各 trace 已有獨立的 hovertemplate，不需統一設定

    return fig


def create_band_filter_comparison(
    frequencies: np.ndarray,
    magnitudes_original: np.ndarray,
    magnitudes_filtered: np.ndarray,
    removed_bands: list,
    title: str = "頻帶過濾前後對比"
) -> go.Figure:
    """建立頻帶過濾前後對比圖

    用於顯示移除特定頻帶後的頻譜變化

    Args:
        frequencies: 頻率陣列 (Hz)
        magnitudes_original: 原始能量陣列 (dB)
        magnitudes_filtered: 過濾後能量陣列 (dB)
        removed_bands: 被移除的頻帶列表
        title: 圖表標題

    Returns:
        go.Figure: Plotly 圖表物件
    """
    fig = go.Figure()

    # 原始頻譜
    fig.add_trace(go.Scatter(
        x=frequencies,
        y=magnitudes_original,
        mode='lines',
        name='原始頻譜',
        line=dict(color='#1f77b4', width=1.5),
        opacity=0.5,
        hovertemplate='原始<br>頻率: %{x:.1f} Hz<br>幅度: %{y:.1f} dB<extra></extra>'
    ))

    # 過濾後頻譜
    fig.add_trace(go.Scatter(
        x=frequencies,
        y=magnitudes_filtered,
        mode='lines',
        name='過濾後頻譜',
        line=dict(color='#2ca02c', width=2),
        hovertemplate='過濾後<br>頻率: %{x:.1f} Hz<br>幅度: %{y:.1f} dB<extra></extra>'
    ))

    # 標示被移除的頻帶區域
    band_ranges = {
        'low_freq': (20, 500, '低頻 (風扇)'),
        'mid_freq': (500, 2000, '中頻'),
        'mid_high_freq': (2000, 6000, '中高頻'),
        'high_freq': (6000, 12000, '高頻 (電感)'),
        'ultra_high_freq': (12000, 20000, '超高頻')
    }

    band_colors = {
        'low_freq': 'rgba(0, 100, 255, 0.15)',
        'mid_freq': 'rgba(0, 200, 100, 0.15)',
        'mid_high_freq': 'rgba(255, 200, 0, 0.15)',
        'high_freq': 'rgba(255, 100, 0, 0.15)',
        'ultra_high_freq': 'rgba(255, 0, 100, 0.15)'
    }

    for band in removed_bands:
        if band in band_ranges:
            low, high, label = band_ranges[band]
            fig.add_vrect(
                x0=low, x1=high,
                fillcolor=band_colors.get(band, 'rgba(128,128,128,0.15)'),
                line=dict(color='red', width=1, dash='dash'),
                annotation_text=f'已移除: {label}',
                annotation_position='top left',
                annotation_font=dict(color='red', size=10)
            )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        xaxis=dict(
            title='頻率 (Hz)',
            type='log',
            range=[np.log10(20), np.log10(20000)],
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='red',
            spikethickness=1,
            spikedash='dot'
        ),
        yaxis=dict(
            title='幅度 (dB)',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='red',
            spikethickness=1,
            spikedash='dot'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        hovermode='closest',
        dragmode='zoom',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=80, b=60)
    )

    return fig


def create_waveform_chart(
    audio: np.ndarray,
    sample_rate: int,
    title: str = "波形圖 (Waveform)"
) -> go.Figure:
    """建立互動式波形圖

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        title: 圖表標題

    Returns:
        go.Figure: Plotly 圖表物件
    """
    # 降低取樣以提升效能 (每秒最多 1000 點)
    max_points_per_sec = 1000
    total_samples = len(audio)
    duration = total_samples / sample_rate
    
    if total_samples > max_points_per_sec * duration:
        step = int(total_samples / (max_points_per_sec * duration))
        audio_display = audio[::step]
        time = np.linspace(0, duration, len(audio_display))
    else:
        time = np.linspace(0, duration, total_samples)
        audio_display = audio

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time,
        y=audio_display,
        mode='lines',
        name='波形',
        line=dict(color='#1f77b4', width=0.5),
        hovertemplate='時間: %{x:.3f}s<br>振幅: %{y:.4f}<extra></extra>'
    ))

    # 標記零線
    fig.add_hline(y=0, line=dict(color='gray', width=1, dash='dash'))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        xaxis=dict(
            title='時間 (秒)',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            showspikes=True,
            spikemode='across',
            spikecolor='red',
            spikethickness=1,
            spikedash='dot'
        ),
        yaxis=dict(
            title='振幅',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            showspikes=True,
            spikemode='across',
            spikecolor='red',
            spikethickness=1,
            spikedash='dot'
        ),
        hovermode='closest',
        dragmode='zoom',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=60, b=60)
    )

    return fig


def create_spectrogram_chart(
    audio: np.ndarray,
    sample_rate: int,
    title: str = "頻譜瀑布圖 (Spectrogram)",
    fmax: int = 20000,
    n_fft: int = 2048,
    hop_length: int = 512,
    use_a_weighting: bool = True,
    z_range: tuple = None,
    calibration_offset: float = 0.0,
    spl_offset: float = 0.0
) -> go.Figure:
    """建立互動式 Spectrogram

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        title: 圖表標題
        fmax: 最大顯示頻率 (Hz)
        n_fft: FFT 視窗大小
        hop_length: 跳躍長度
        use_a_weighting: 是否套用 A-weighting
        z_range: 手動設定色彩範圍 (z_min, z_max)，None 表示自動
        calibration_offset: 麥克風校準偏移 (dB)
        spl_offset: dB SPL 絕對模式偏移 (dB)

    Returns:
        go.Figure: Plotly 圖表物件
    """
    from scipy.signal import spectrogram as scipy_spectrogram
    from core.noise_level import apply_a_weighting

    
    # 如果啟用 A-weighting，先對音訊套用
    if use_a_weighting:
        audio_processed = apply_a_weighting(audio, sample_rate)
    else:
        audio_processed = audio
    
    # 計算 Spectrogram
    frequencies, times, Sxx = scipy_spectrogram(
        audio_processed, fs=sample_rate,
        nperseg=n_fft, noverlap=n_fft - hop_length
    )
    
    # 轉換為 dB (相對功率)
    # 使用 10*log10 因為 Sxx 是功率譜密度
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # 套用校準偏移（麥克風校準 + SPL 模式偏移）
    total_offset = calibration_offset + spl_offset
    Sxx_db = Sxx_db + total_offset
    
    # 限制頻率範圍
    freq_mask = frequencies <= min(fmax, sample_rate / 2)
    frequencies = frequencies[freq_mask]
    Sxx_db = Sxx_db[freq_mask, :]
    
    # 動態單位標籤
    unit_label = "dB(A)" if use_a_weighting else "dB"
    if spl_offset > 0:
        unit_label += " SPL"

    fig = go.Figure()

    # 色彩範圍設定
    heatmap_kwargs = dict(
        x=times,
        y=frequencies,
        z=Sxx_db,
        colorscale='Viridis',
        colorbar=dict(
            title=dict(text=f'幅度 ({unit_label})', side='right')
        ),
        hovertemplate=f'時間: %{{x:.2f}}s<br>頻率: %{{y:.0f}} Hz<br>幅度: %{{z:.1f}} {unit_label}<extra></extra>'
    )
    
    # 只在手動模式下設定 zmin/zmax，否則讓 Plotly 自動計算（與播放器一致）
    if z_range is not None:
        heatmap_kwargs['zmin'] = z_range[0]
        heatmap_kwargs['zmax'] = z_range[1]

    fig.add_trace(go.Heatmap(**heatmap_kwargs))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        xaxis=dict(
            title='時間 (秒)',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)'
        ),
        yaxis=dict(
            title='頻率 (Hz)',
            type='log',
            range=[np.log10(20), np.log10(min(fmax, sample_rate/2))],
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)'
        ),
        dragmode='zoom',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=100, t=60, b=60)
    )

    return fig


def create_a_weighting_chart(
    sample_rate: int = 48000,
    title: str = "A-weighting 頻率響應曲線"
) -> go.Figure:
    """建立 A-weighting 曲線圖

    顯示 A-weighting 濾波器的頻率響應，展示人耳對不同頻率的敏感度。

    Args:
        sample_rate: 取樣率 (Hz)
        title: 圖表標題

    Returns:
        go.Figure: Plotly 圖表物件
    """
    # A-weighting 計算公式 (IEC 61672-1)
    frequencies = np.logspace(np.log10(20), np.log10(20000), 500)
    
    f = frequencies
    f2 = f ** 2
    
    # A-weighting 係數
    ra = (12194**2 * f2**2) / (
        (f2 + 20.6**2) *
        np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) *
        (f2 + 12194**2)
    )
    
    # 轉換為 dB (相對於 1kHz)
    a_weight_db = 20 * np.log10(ra) + 2.0  # 調整使 1kHz 為 0 dB

    fig = go.Figure()

    # A-weighting 曲線
    fig.add_trace(go.Scatter(
        x=frequencies,
        y=a_weight_db,
        mode='lines',
        name='A-weighting',
        line=dict(color='#e74c3c', width=2),
        hovertemplate='頻率: %{x:.0f} Hz<br>加權: %{y:.1f} dB<extra></extra>'
    ))

    # 標記 1kHz 參考點
    fig.add_vline(
        x=1000,
        line=dict(color='gray', width=1, dash='dash'),
        annotation_text='1 kHz (參考)',
        annotation_position='top'
    )
    
    fig.add_hline(
        y=0,
        line=dict(color='gray', width=1, dash='dash')
    )

    # 標記人耳敏感區域
    fig.add_vrect(
        x0=2000, x1=5000,
        fillcolor='rgba(46, 204, 113, 0.1)',
        line_width=0,
        annotation_text='人耳敏感區 (2-5kHz)',
        annotation_position='top left',
        annotation_font=dict(color='green', size=10)
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        xaxis=dict(
            title='頻率 (Hz)',
            type='log',
            range=[np.log10(20), np.log10(20000)],
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            showspikes=True,
            spikemode='across',
            spikecolor='red',
            spikethickness=1,
            spikedash='dot'
        ),
        yaxis=dict(
            title='加權 (dB)',
            range=[-50, 5],
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            showspikes=True,
            spikemode='across',
            spikecolor='red',
            spikethickness=1,
            spikedash='dot'
        ),
        hovermode='closest',
        dragmode='zoom',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=60, b=60)
    )

    return fig


def create_octave_band_chart(
    audio: np.ndarray,
    sample_rate: int,
    title: str = "1/3 倍頻程分析",
    use_a_weighting: bool = True,
    calibration_offset: float = 0.0
) -> go.Figure:
    """建立 1/3 倍頻程分析圖

    使用 IEC 61260-1:2014 標準的 Butterworth 濾波器組實現。
    與 HEAD acoustics ArtemiS SUITE 的 "1/n Octave (Filter)" 方法一致。

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        title: 圖表標題
        use_a_weighting: 是否套用 A-weighting (預設 True)
        calibration_offset: 校準偏移值 (dB)

    Returns:
        go.Figure: Plotly 圖表物件
    """
    # 使用 compute_octave_bands 函數 (IEC 61260 濾波器法)
    octave_data = compute_octave_bands(audio, sample_rate, use_a_weighting=use_a_weighting)
    
    nominal_freqs = octave_data["nominal_freqs"]
    # 套用校準偏移
    band_levels = [level + calibration_offset for level in octave_data["band_levels"]]
    method = octave_data.get("method", "IEC 61260-1:2014 Filter Bank")
    
    unit_label = "dB(A)" if use_a_weighting else "dB"
    
    # 建構 X 軸標籤列表
    x_labels = [f'{int(f)}' if f < 1000 else f'{f/1000:.1f}k' for f in nominal_freqs]
    
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=x_labels,
        y=band_levels,
        marker_color='#2ecc71',
        marker_line=dict(color='#1e8449', width=1),
        hovertemplate='頻率: %{customdata:.0f} Hz<br>Level: %{y:.1f} ' + unit_label + '<extra></extra>',
        customdata=nominal_freqs,
        # 只顯示 > -50 dB 的數值，避免畫面過於雜亂
        text=[f"{y:.1f}" if y > -50 else "" for y in band_levels], 
        textposition='auto',
        textfont=dict(size=10)
    ))
    
    # 圖表標題（包含校準標記）
    cal_note = f" (Cal: {calibration_offset:+.1f}dB)" if calibration_offset != 0 else ""
    
    fig.update_layout(
        title=dict(text=f"{title} ({unit_label})", font=dict(size=16, color='#333')),
        annotations=[
            dict(
                text=f"Sample Rate: {sample_rate:.0f}Hz | Method: {method}{cal_note}",
                xref="paper", yref="paper",
                x=1.0, y=1.05,
                showarrow=False,
                font=dict(size=10, color="gray")
            )
        ],
        xaxis=dict(
            title='中心頻率 (Hz)',
            tickangle=-45,
            tickmode='array',
            tickvals=list(range(len(x_labels))),
            ticktext=x_labels,
            type='category'
        ),
        yaxis=dict(
            title=f'L(A) {unit_label}' if use_a_weighting else f'Level ({unit_label})',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)'
        ),
        dragmode='zoom',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=80, b=100),
        bargap=0.15
    )

    return fig


def create_waterfall_3d_chart(
    audio: np.ndarray,
    sample_rate: int,
    title: str = "3D Waterfall 頻譜圖",
    fmax: int = 10000,
    n_fft: int = 2048,
    hop_length: int = 1024,
    n_time_slices: int = 50,
    use_a_weighting: bool = True,
    calibration_offset: float = 0.0,
    spl_offset: float = 0.0
) -> go.Figure:
    """建立 3D Waterfall 頻譜圖

    3D 視覺化頻譜隨時間的變化。

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        title: 圖表標題
        fmax: 最大顯示頻率 (Hz)
        n_fft: FFT 視窗大小
        hop_length: 跳躍長度
        n_time_slices: 時間切片數量
        use_a_weighting: 是否套用 A-weighting
        calibration_offset: 麥克風校準偏移 (dB)
        spl_offset: dB SPL 模式偏移 (dB)

    Returns:
        go.Figure: Plotly 圖表物件
    """
    from scipy.signal import spectrogram as scipy_spectrogram
    from core.noise_level import apply_a_weighting
    
    # 如果啟用 A-weighting，先對音訊套用
    if use_a_weighting:
        audio_processed = apply_a_weighting(audio, sample_rate)
    else:
        audio_processed = audio
    
    # 計算 Spectrogram
    frequencies, times, Sxx = scipy_spectrogram(
        audio_processed, fs=sample_rate,
        nperseg=n_fft, noverlap=n_fft - hop_length
    )
    
    # 轉換為 dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # 套用校準偏移（麥克風校準 + SPL 模式偏移）
    total_offset = calibration_offset + spl_offset
    Sxx_db = Sxx_db + total_offset
    
    # 限制頻率範圍
    freq_mask = frequencies <= min(fmax, sample_rate / 2)
    frequencies = frequencies[freq_mask]
    Sxx_db = Sxx_db[freq_mask, :]
    
    # 降低時間解析度以提升效能
    time_step = max(1, len(times) // n_time_slices)
    times_sub = times[::time_step]
    Sxx_db_sub = Sxx_db[:, ::time_step]
    
    # 動態單位標籤
    unit_label = "dB(A)" if use_a_weighting else "dB"
    if spl_offset > 0:
        unit_label += " SPL"

    fig = go.Figure()

    # 3D 曲面
    fig.add_trace(go.Surface(
        x=times_sub,
        y=frequencies,
        z=Sxx_db_sub,
        colorscale='Viridis',
        colorbar=dict(
            title=dict(text=f'幅度 ({unit_label})', side='right'),
            len=0.5
        ),
        hovertemplate=f'時間: %{{x:.2f}}s<br>頻率: %{{y:.0f}} Hz<br>幅度: %{{z:.1f}} {unit_label}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        scene=dict(
            xaxis=dict(title='時間 (秒)'),
            yaxis=dict(title='頻率 (Hz)', type='log'),
            zaxis=dict(title=f'幅度 ({unit_label})'),
            camera=dict(
                eye=dict(x=1.5, y=-1.5, z=0.8)
            )
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        height=600
    )

    return fig


def create_combined_analysis_chart(
    audio: np.ndarray,
    sample_rate: int,
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    title: str = "綜合分析視圖",
    calibration_offset: float = 0.0,
    spl_offset: float = 0.0,
    z_range: Tuple[float, float] = None,
    use_a_weighting: bool = True,
    smooth_window: int = 10
) -> go.Figure:
    """建立綜合分析視圖 (2x2 子圖)

    同時顯示 Level vs Time、FFT 頻譜、Spectrogram 和 1/3 倍頻程。

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        frequencies: FFT 頻率陣列
        magnitudes: FFT 幅度陣列
        title: 圖表標題
        calibration_offset: 校準偏移 (dB)
        spl_offset: Spectrogram SPL 偏移 (dB)
        z_range: Spectrogram 顯示範圍 (min_db, max_db)
        use_a_weighting: 是否套用 A-weighting
        smooth_window: Level vs Time 平滑視窗大小

    Returns:
        go.Figure: Plotly 圖表物件
    """
    from scipy.signal import spectrogram as scipy_spectrogram
    from core.band_analyzer import compute_octave_bands
    from core.noise_level import apply_a_weighting as apply_a_wt
    
    # 建立 2x2 子圖 - 增加間距
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Level vs Time', 'FFT 頻譜', 'Spectrogram', '1/3 倍頻程'),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "heatmap"}, {"type": "bar"}]
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.12
    )

    # 1. Level vs Time
    from core.noise_level import calculate_frame_levels
    
    frame_length = 4096 
    
    # 計算每幀的音壓級 (含 A-weighting)
    frame_levels = calculate_frame_levels(audio, sample_rate, frame_length, apply_weighting=True)
    
    # 套用校準偏移
    frame_levels = frame_levels + calibration_offset
    
    # 計算時間軸
    n_frames = len(frame_levels)
    times_lv = np.arange(n_frames) * frame_length / sample_rate
    
    # 平滑處理
    if smooth_window > 1:
        kernel = np.ones(smooth_window) / smooth_window
        frame_levels = np.convolve(frame_levels, kernel, mode='same')
    
    # 降低資料點數量
    step = max(1, len(times_lv) // 500)
    times_lv = times_lv[::step]
    levels_lv = frame_levels[::step]
    
    fig.add_trace(
        go.Scatter(
            x=times_lv, y=levels_lv,
            mode='lines', name='Level',
            line=dict(color='#2ca02c', width=1),
            hovertemplate='<b>%{x:.1f}s</b> | %{y:.1f} dBA<extra></extra>'
        ),
        row=1, col=1
    )

    # 2. FFT 頻譜 (需套用校準偏移)
    # 這裡的 magnitudes 已經是 dB 了，直接加上 offset
    calibrated_magnitudes = magnitudes + calibration_offset
    
    fig.add_trace(
        go.Scatter(
            x=frequencies, y=calibrated_magnitudes,
            mode='lines', name='頻譜',
            line=dict(color='#ff7f0e', width=1),
            hovertemplate='<b>%{x:.1f} Hz</b> | %{y:.1f} dB<extra></extra>'
        ),
        row=1, col=2
    )

    # 3. Spectrogram (需套用 calibration_offset + spl_offset)
    # 如果啟用 A-weighting，先對音訊套用
    if use_a_weighting:
        audio_spec = apply_a_wt(audio, sample_rate)
    else:
        audio_spec = audio

    # 使用與 create_spectrogram_chart 相同的參數
    n_fft = 2048
    hop_length = 512
    freq_spec, times_spec, Sxx = scipy_spectrogram(
        audio_spec, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # 套用偏移
    total_offset = calibration_offset + spl_offset
    Sxx_db += total_offset
    
    freq_mask = freq_spec <= 10000
    
    zmin, zmax = None, None
    if z_range:
        zmin, zmax = z_range
        
    fig.add_trace(
        go.Heatmap(
            x=times_spec,
            y=freq_spec[freq_mask],
            z=Sxx_db[freq_mask, :],
            colorscale='Viridis',
            showscale=False,
            zmin=zmin,
            zmax=zmax,
            hovertemplate='時間: %{x:.2f}s<br>頻率: %{y:.0f}Hz<br>強度: %{z:.1f} dB<extra></extra>'
        ),
        row=2, col=1
    )

    # 4. 1/3 倍頻程分析
    octave_result = compute_octave_bands(audio, sample_rate)
    center_freqs = octave_result['nominal_freqs']
    band_levels = octave_result['band_levels']
    
    # 過濾掉無效的頻帶（-120 dB 表示無法計算），並加上校準偏移
    valid_indices = [i for i, level in enumerate(band_levels) if level > -100]
    center_freqs = [center_freqs[i] for i in valid_indices]
    
    # 加上校準偏移
    band_levels = [band_levels[i] + calibration_offset for i in valid_indices]
    
    # 生成頻率標籤（簡化高頻標籤）
    freq_labels = []
    for f in center_freqs:
        if f >= 1000:
            freq_labels.append(f"{f/1000:.0f}k" if f % 1000 == 0 else f"{f/1000:.1f}k")
        else:
            freq_labels.append(f"{int(f)}")
    
    # 使用漸層色彩
    n_bands = len(center_freqs)
    colors = [f'hsl({i * 360 / n_bands}, 70%, 50%)' for i in range(n_bands)]
    
    fig.add_trace(
        go.Bar(
            x=list(range(len(freq_labels))),  # 使用數字索引
            y=band_levels,
            marker_color=colors,
            text=freq_labels,  # 顯示頻率標籤
            textposition='none',
            hovertemplate='<b>%{text}</b><br>%{y:.1f} dB(A)<extra></extra>'
        ),
        row=2, col=2
    )

    # 更新軸設定
    
    # 1. Level vs Time (Row 1, Col 1)
    fig.update_xaxes(
        title_text='時間 (秒)', 
        row=1, col=1,
        matches='x3'  # 與 Spectrogram (Row 2, Col 1) 共用 X 軸縮放
    )
    fig.update_yaxes(title_text='音壓級 (dBA)', row=1, col=1, fixedrange=False)
    
    # 2. FFT 頻譜 (Row 1, Col 2)
    fig.update_xaxes(
        title_text='頻率 (Hz)', 
        type='log', 
        row=1, col=2,
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        spikecolor='red',
        spikethickness=1,
        spikedash='dot',
        fixedrange=False
    )
    fig.update_yaxes(title_text='幅度 (dB)', row=1, col=2, fixedrange=False)
    
    # 3. Spectrogram (Row 2, Col 1)
    fig.update_xaxes(
        title_text='時間 (秒)', 
        row=2, col=1,
        matches='x1'  # 與 Level vs Time (Row 1, Col 1) 共用 X 軸縮放
    )
    
    # 設定 Y 軸為對數刻度，並設定範圍
    log_range = [np.log10(20), np.log10(sample_rate / 2)]
    
    fig.update_yaxes(
        title_text='頻率 (Hz)', 
        type='log',
        range=log_range,
        row=2, col=1, 
        fixedrange=False
    )
    
    # 4. 1/3 倍頻程 (Row 2, Col 2) - 維持固定 (不可縮放)
    fig.update_xaxes(
        title_text='中心頻率', 
        tickangle=-45,
        tickmode='array',
        tickvals=list(range(len(freq_labels))),
        ticktext=freq_labels,
        type='category',
        row=2, col=2,
        fixedrange=True
    )
    fig.update_yaxes(title_text='能量 (dB(A))', row=2, col=2, fixedrange=True)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        showlegend=False,
        dragmode='zoom',  # 預設啟用縮放工具
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=80, b=60),
        height=750
    )

    return fig


def create_level_vs_time_chart(
    audio: np.ndarray,
    sample_rate: int,
    frame_length: int = 4096,
    smooth_window: int = 5,
    title: str = "Level vs Time (dBA)",
    leq: float = None,
    calibration_offset: float = 0.0,
    use_a_weighting: bool = True  # 新增參數
) -> go.Figure:
    """建立時間對應音壓級圖表
    
    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        frame_length: 分析幀長度
        smooth_window: 滑動平均窗口大小 (用於平滑曲線)
        title: 圖表標題
        leq: 總 Leq 值（已不再使用，保留參數兼容性）
        calibration_offset: 校準偏移 (dB)
        use_a_weighting: 是否套用 A-weighting
    
    Returns:
        go.Figure: Plotly 圖表物件
    """
    from core.noise_level import calculate_frame_levels
    
    # 動態更新標題
    if use_a_weighting:
        ylabel = "L[A] dB(SPL)"
        if "(dBA)" not in title and "(dB SPL)" not in title:
           title = f"{title} (dBA)"
    else:
        ylabel = "L[Z] dB(SPL)"
        title = title.replace("(dBA)", "(dB SPL)")
    
    # DEBUG: 顯示校準偏移
    if calibration_offset != 0:
        title += f" (Cal: +{calibration_offset:.1f}dB)"
    
    # 計算每幀的音壓級
    frame_levels = calculate_frame_levels(audio, sample_rate, frame_length, apply_weighting=use_a_weighting)
    
    # 套用校準偏移
    frame_levels = frame_levels + calibration_offset
    
    # 計算時間軸 (使用幀長度，因為 calculate_frame_levels 不重疊)
    n_frames = len(frame_levels)
    times = np.arange(n_frames) * frame_length / sample_rate
    
    # 滑動平均平滑處理
    if smooth_window > 1 and len(frame_levels) > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        frame_levels_smooth = np.convolve(frame_levels, kernel, mode='same')
        # 處理邊緣效應
        half_win = smooth_window // 2
        frame_levels_smooth[:half_win] = frame_levels[:half_win]
        frame_levels_smooth[-half_win:] = frame_levels[-half_win:]
    else:
        frame_levels_smooth = frame_levels
    
    # 計算 Y 軸範圍（自適應居中）
    valid_levels = frame_levels_smooth[np.isfinite(frame_levels_smooth)]
    if len(valid_levels) > 0:
        y_min = np.min(valid_levels)
        y_max = np.max(valid_levels)
        y_range = y_max - y_min
        # 增加上下 20% 的邊距
        y_margin = max(y_range * 0.2, 2)  # 至少 2 dB 邊距
        y_axis_min = y_min - y_margin
        y_axis_max = y_max + y_margin
    else:
        y_axis_min = 0
        y_axis_max = 100
    
    fig = go.Figure()
    
    # 繪製音壓級曲線（綠色，無填充）
    fig.add_trace(go.Scatter(
        x=times,
        y=frame_levels_smooth,
        mode='lines',
        name=ylabel,
        line=dict(color='#2ca02c', width=1.2),
        hovertemplate=f'時間: %{{x:.2f}}s<br>{ylabel}: %{{y:.1f}} dB<extra></extra>'
    ))
    
    # 設定布局（Y 軸自適應範圍）
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        xaxis=dict(
            title='時間 (s)',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='red',
            spikethickness=1,
            spikedash='dot'
        ),
        yaxis=dict(
            title=ylabel,
            range=[y_axis_min, y_axis_max],
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            showspikes=False
        ),
        hovermode='x',  # 在任意 Y 位置都可觸發 hover
        hoverlabel=dict(
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(100,100,100,0.3)',
            font_size=11
        ),
        dragmode='zoom',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=60, b=60)
    )
    
    # 設定緊湊的 hover 資訊
    fig.update_traces(
        hovertemplate='<b>%{x:.1f}s</b> | %{y:.1f} dB<extra></extra>'
    )
    
    return fig


def create_spectrum_with_leq_line(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    leq: float,
    title: str = "頻譜圖",
    log_scale: bool = True,
    freq_range: Tuple[float, float] = None,
    ylabel: str = "幅度 (dB)"
) -> go.Figure:
    """建立帶有 Leq 參考線的頻譜圖
    
    Args:
        frequencies: 頻率陣列 (Hz)
        magnitudes: 能量陣列 (dB)
        leq: 總 Leq 值
        title: 圖表標題
        log_scale: 是否使用對數頻率軸
        freq_range: 顯示的頻率範圍
        ylabel: Y 軸標籤
    
    Returns:
        go.Figure: Plotly 圖表物件
    """
    fig = go.Figure()

    # 新增頻譜線
    fig.add_trace(go.Scatter(
        x=frequencies,
        y=magnitudes,
        mode='lines',
        name='頻譜',
        line=dict(color='#1f77b4', width=1),
        hovertemplate='頻率: %{x:.1f} Hz<br>幅度: %{y:.1f} dB<extra></extra>'
    ))
    
    # 添加 Leq 參考線
    fig.add_hline(
        y=leq,
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        annotation_text=f'Leq: {leq:.1f} dB (總能量)',
        annotation_position='top right',
        annotation_font=dict(color='#ff7f0e', size=11)
    )

    # 設定頻率範圍
    if freq_range is None:
        freq_range = (max(20, frequencies[0]), min(20000, frequencies[-1]))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        xaxis=dict(
            title='頻率 (Hz)',
            type='log' if log_scale else 'linear',
            range=[np.log10(freq_range[0]), np.log10(freq_range[1])] if log_scale else list(freq_range),
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            tickformat='.0f',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='red',
            spikethickness=1,
            spikedash='dot'
        ),
        yaxis=dict(
            title=ylabel,
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='red',
            spikethickness=1,
            spikedash='dot'
        ),
        hovermode='closest',
        dragmode='zoom',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=60, b=60)
    )

    return fig
