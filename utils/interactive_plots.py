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


def create_interactive_spectrum(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    title: str = "頻譜圖",
    log_scale: bool = True,
    freq_range: Tuple[float, float] = None,
    show_grid: bool = True
) -> go.Figure:
    """建立互動式頻譜圖

    Args:
        frequencies: 頻率陣列 (Hz)
        magnitudes: 能量陣列 (dB)
        title: 圖表標題
        log_scale: 是否使用對數頻率軸
        freq_range: 顯示的頻率範圍
        show_grid: 是否顯示網格

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

    # 設定布局 - 使用 closest 模式讓十字座標完整顯示
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
            spikesnap='cursor',
            spikecolor='red',
            spikethickness=1,
            spikedash='dot'
        ),
        yaxis=dict(
            title='幅度 (dB)',
            showgrid=show_grid,
            gridcolor='rgba(128, 128, 128, 0.3)',
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikecolor='red',
            spikethickness=1,
            spikedash='dot'
        ),
        hovermode='closest',  # 改用 closest 模式讓十字座標完整顯示
        dragmode='zoom',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=60, b=60),
        modebar=dict(
            orientation='h',
            bgcolor='rgba(255,255,255,0.7)'
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
    title: str = "Discrete Tone 檢測結果"
) -> go.Figure:
    """建立 Discrete Tone 視覺化圖表

    在頻譜圖上標記偵測到的 Discrete Tone 位置

    Args:
        frequencies: 頻率陣列 (Hz)
        magnitudes: 能量陣列 (dB)
        tones: 超過門檻的 Tone 列表
        all_candidates: 所有候選峰值列表
        title: 圖表標題

    Returns:
        go.Figure: Plotly 圖表物件
    """
    fig = go.Figure()

    # 繪製頻譜基線
    fig.add_trace(go.Scatter(
        x=frequencies,
        y=magnitudes,
        mode='lines',
        name='頻譜',
        line=dict(color='#1f77b4', width=1),
        hovertemplate='頻率: %{x:.1f} Hz<br>幅度: %{y:.1f} dB<extra></extra>'
    ))

    # 標記所有候選峰值 (灰色三角形)
    if all_candidates:
        candidate_freqs = [t.get('frequency', 0) for t in all_candidates if not t.get('exceeds_threshold', False)]
        candidate_mags = [t.get('magnitude_db', 0) for t in all_candidates if not t.get('exceeds_threshold', False)]
        candidate_proms = [t.get('prominence', 0) for t in all_candidates if not t.get('exceeds_threshold', False)]
        
        if candidate_freqs:
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
                hovertemplate='候選峰值<br>頻率: %{x:.1f} Hz<br>幅度: %{y:.1f} dB<br>突出量: %{customdata:.1f} dB<extra></extra>',
                customdata=candidate_proms
            ))

    # 標記超過門檻的 Discrete Tone (紅色星形)
    if tones:
        tone_freqs = [t.get('frequency', 0) for t in tones]
        tone_mags = [t.get('magnitude_db', 0) for t in tones]
        tone_proms = [t.get('prominence', 0) for t in tones]
        tone_thresholds = [t.get('threshold', 0) for t in tones]
        tone_bands = [t.get('band', 'N/A') for t in tones]
        
        hover_texts = [
            f"⚠️ Discrete Tone<br>"
            f"頻率: {f:.1f} Hz<br>"
            f"幅度: {m:.1f} dB<br>"
            f"突出量: {p:.1f} dB<br>"
            f"門檻: {th:.1f} dB<br>"
            f"頻帶: {b}"
            for f, m, p, th, b in zip(tone_freqs, tone_mags, tone_proms, tone_thresholds, tone_bands)
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

    # ECMA-74 門檻區域標示
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
            name=f'{label}',
            marker=dict(size=10, color=colors[i].replace('0.05', '0.3')),
            showlegend=True
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        xaxis=dict(
            title='頻率 (Hz)',
            type='log',
            range=[np.log10(50), np.log10(15000)],
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
    hop_length: int = 512
) -> go.Figure:
    """建立互動式 Spectrogram

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        title: 圖表標題
        fmax: 最大顯示頻率 (Hz)
        n_fft: FFT 視窗大小
        hop_length: 跳躍長度

    Returns:
        go.Figure: Plotly 圖表物件
    """
    from scipy.signal import spectrogram as scipy_spectrogram
    
    # 計算 Spectrogram
    frequencies, times, Sxx = scipy_spectrogram(
        audio, fs=sample_rate,
        nperseg=n_fft, noverlap=n_fft - hop_length
    )
    
    # 轉換為 dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # 限制頻率範圍
    freq_mask = frequencies <= min(fmax, sample_rate / 2)
    frequencies = frequencies[freq_mask]
    Sxx_db = Sxx_db[freq_mask, :]

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        x=times,
        y=frequencies,
        z=Sxx_db,
        colorscale='Viridis',
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
    title: str = "1/3 倍頻程分析"
) -> go.Figure:
    """建立 1/3 倍頻程分析圖

    依據 ISO 標準頻帶分析能量分布。

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        title: 圖表標題

    Returns:
        go.Figure: Plotly 圖表物件
    """
    from scipy.signal import butter, sosfilt
    
    # 1/3 倍頻程中心頻率 (ISO 規範)
    center_freqs = [
        25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200,
        250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000,
        2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500, 16000, 20000
    ]
    
    # 過濾超出 Nyquist 頻率的頻帶
    nyquist = sample_rate / 2
    center_freqs = [f for f in center_freqs if f < nyquist * 0.9]
    
    band_levels = []
    valid_freqs = []
    
    for fc in center_freqs:
        # 1/3 倍頻程頻寬
        factor = 2 ** (1/6)  # 1/3 octave
        f_low = fc / factor
        f_high = fc * factor
        
        # 確保在有效範圍內
        if f_high >= nyquist:
            continue
        if f_low < 20:
            f_low = 20
            
        try:
            # 帶通濾波
            sos = butter(4, [f_low, f_high], btype='band', fs=sample_rate, output='sos')
            filtered = sosfilt(sos, audio)
            
            # 計算 RMS 能量
            rms = np.sqrt(np.mean(filtered ** 2))
            level_db = 20 * np.log10(rms + 1e-10)
            
            band_levels.append(level_db)
            valid_freqs.append(fc)
        except:
            pass

    fig = go.Figure()

    # 長條圖
    colors = ['#3498db' if level > -60 else '#95a5a6' for level in band_levels]
    
    fig.add_trace(go.Bar(
        x=[f'{f:.0f}' if f < 1000 else f'{f/1000:.1f}k' for f in valid_freqs],
        y=band_levels,
        marker_color=colors,
        marker_line=dict(color='#2c3e50', width=1),
        hovertemplate='頻率: %{customdata:.0f} Hz<br>能量: %{y:.1f} dB<extra></extra>',
        customdata=valid_freqs
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        xaxis=dict(
            title='中心頻率 (Hz)',
            tickangle=-45
        ),
        yaxis=dict(
            title='能量 (dB)',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.3)'
        ),
        dragmode='zoom',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=60, b=100),
        bargap=0.2
    )

    return fig


def create_waterfall_3d_chart(
    audio: np.ndarray,
    sample_rate: int,
    title: str = "3D Waterfall 頻譜圖",
    fmax: int = 10000,
    n_fft: int = 2048,
    hop_length: int = 1024,
    n_time_slices: int = 50
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

    Returns:
        go.Figure: Plotly 圖表物件
    """
    from scipy.signal import spectrogram as scipy_spectrogram
    
    # 計算 Spectrogram
    frequencies, times, Sxx = scipy_spectrogram(
        audio, fs=sample_rate,
        nperseg=n_fft, noverlap=n_fft - hop_length
    )
    
    # 轉換為 dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # 限制頻率範圍
    freq_mask = frequencies <= min(fmax, sample_rate / 2)
    frequencies = frequencies[freq_mask]
    Sxx_db = Sxx_db[freq_mask, :]
    
    # 降低時間解析度以提升效能
    time_step = max(1, len(times) // n_time_slices)
    times_sub = times[::time_step]
    Sxx_db_sub = Sxx_db[:, ::time_step]

    fig = go.Figure()

    # 3D 曲面
    fig.add_trace(go.Surface(
        x=times_sub,
        y=frequencies,
        z=Sxx_db_sub,
        colorscale='Viridis',
        colorbar=dict(
            title=dict(text='幅度 (dB)', side='right'),
            len=0.5
        ),
        hovertemplate='時間: %{x:.2f}s<br>頻率: %{y:.0f} Hz<br>幅度: %{z:.1f} dB<extra></extra>'
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        scene=dict(
            xaxis=dict(title='時間 (秒)'),
            yaxis=dict(title='頻率 (Hz)', type='log'),
            zaxis=dict(title='幅度 (dB)'),
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
    title: str = "綜合分析視圖"
) -> go.Figure:
    """建立綜合分析視圖 (2x2 子圖)

    同時顯示波形、頻譜、Spectrogram 和 1/3 倍頻程。

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        frequencies: FFT 頻率陣列
        magnitudes: FFT 幅度陣列
        title: 圖表標題

    Returns:
        go.Figure: Plotly 圖表物件
    """
    from scipy.signal import spectrogram as scipy_spectrogram
    
    # 建立 2x2 子圖
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('波形圖', 'FFT 頻譜', 'Spectrogram', '頻帶能量'),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "heatmap"}, {"type": "bar"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.08
    )

    # 1. 波形圖 (降低取樣)
    duration = len(audio) / sample_rate
    step = max(1, len(audio) // 2000)
    time = np.linspace(0, duration, len(audio[::step]))
    
    fig.add_trace(
        go.Scatter(
            x=time, y=audio[::step],
            mode='lines', name='波形',
            line=dict(color='#1f77b4', width=0.5)
        ),
        row=1, col=1
    )

    # 2. FFT 頻譜
    fig.add_trace(
        go.Scatter(
            x=frequencies, y=magnitudes,
            mode='lines', name='頻譜',
            line=dict(color='#ff7f0e', width=1)
        ),
        row=1, col=2
    )

    # 3. Spectrogram
    freq_spec, times_spec, Sxx = scipy_spectrogram(
        audio, fs=sample_rate, nperseg=1024, noverlap=512
    )
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    freq_mask = freq_spec <= 10000
    
    fig.add_trace(
        go.Heatmap(
            x=times_spec,
            y=freq_spec[freq_mask],
            z=Sxx_db[freq_mask, :],
            colorscale='Viridis',
            showscale=False
        ),
        row=2, col=1
    )

    # 4. 簡化頻帶能量
    bands = ['低頻', '中頻', '中高頻', '高頻', '超高頻']
    band_ranges = [(20, 500), (500, 2000), (2000, 6000), (6000, 12000), (12000, 20000)]
    energies = []
    
    for low, high in band_ranges:
        mask = (frequencies >= low) & (frequencies <= high)
        if np.any(mask):
            # 將 dB 轉回線性計算平均
            linear = 10 ** (magnitudes[mask] / 20)
            avg_db = 20 * np.log10(np.mean(linear) + 1e-10)
            energies.append(avg_db)
        else:
            energies.append(-100)

    fig.add_trace(
        go.Bar(
            x=bands, y=energies,
            marker_color=['#3498db', '#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6']
        ),
        row=2, col=2
    )

    # 更新軸設定
    fig.update_xaxes(title_text='時間 (秒)', row=1, col=1)
    fig.update_yaxes(title_text='振幅', row=1, col=1)
    
    fig.update_xaxes(title_text='頻率 (Hz)', type='log', row=1, col=2)
    fig.update_yaxes(title_text='幅度 (dB)', row=1, col=2)
    
    fig.update_xaxes(title_text='時間 (秒)', row=2, col=1)
    fig.update_yaxes(title_text='頻率 (Hz)', row=2, col=1)
    
    fig.update_xaxes(title_text='頻帶', row=2, col=2)
    fig.update_yaxes(title_text='能量 (dB)', row=2, col=2)

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#333')),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=40, t=80, b=60),
        height=700
    )

    return fig

