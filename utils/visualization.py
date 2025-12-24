# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - 視覺化模組

功能:
- AUD-002: 頻譜圖生成
- AUD-007: 頻譜瀑布圖 (Spectrogram) 生成
- 濾波前後對比圖
- A-weighting 響應曲線圖
"""

from typing import Optional, Tuple, List
import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非互動後端
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# 從 config 導入設定
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from app.config import config

# 設定中文字型支援
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_spectrum(
    frequencies: np.ndarray,
    magnitudes: np.ndarray,
    title: str = "頻譜圖",
    figsize: Tuple[int, int] = None,
    log_scale: bool = True,
    freq_range: Tuple[float, float] = None
) -> bytes:
    """繪製頻譜圖並回傳 PNG bytes

    Args:
        frequencies: 頻率陣列 (Hz)
        magnitudes: 能量陣列 (dB)
        title: 圖表標題
        figsize: 圖表尺寸 (英吋)
        log_scale: 是否使用對數頻率軸
        freq_range: 顯示的頻率範圍 (Hz)

    Returns:
        bytes: PNG 圖片的 bytes 資料
    """
    if figsize is None:
        figsize = config.visualization.spectrum_figsize

    fig, ax = plt.subplots(figsize=figsize)

    # 繪製頻譜
    if log_scale:
        ax.semilogx(frequencies, magnitudes, 'b-', linewidth=0.8)
    else:
        ax.plot(frequencies, magnitudes, 'b-', linewidth=0.8)

    # 設定軸標籤
    ax.set_xlabel('頻率 (Hz)', fontsize=12)
    ax.set_ylabel('幅度 (dB)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 設定頻率範圍
    if freq_range:
        ax.set_xlim(freq_range)
    elif len(frequencies) > 0:
        ax.set_xlim(max(20, frequencies[0]), min(20000, frequencies[-1]))

    # 設定網格
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # 調整布局
    plt.tight_layout()

    # 轉換為 bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    png_bytes = buf.getvalue()
    plt.close(fig)

    return png_bytes


def plot_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    title: str = "頻譜瀑布圖 (Spectrogram)",
    figsize: Tuple[int, int] = None,
    fmax: int = None,
    cmap: str = None
) -> bytes:
    """繪製 Spectrogram 並回傳 PNG bytes

    Args:
        audio: 音訊資料
        sample_rate: 取樣率 (Hz)
        title: 圖表標題
        figsize: 圖表尺寸 (英吋)
        fmax: 最大顯示頻率 (Hz)
        cmap: 顏色映射

    Returns:
        bytes: PNG 圖片的 bytes 資料
    """
    if figsize is None:
        figsize = config.visualization.spectrogram_figsize
    if fmax is None:
        fmax = config.visualization.fmax
    if cmap is None:
        cmap = config.visualization.cmap

    # === 長音訊優化（超過 10 分鐘）- 增加 hop_length ===
    TEN_MINUTES_SAMPLES = sample_rate * 600
    audio_len = len(audio)
    
    # 動態調整 hop_length
    hop_length = config.fft.hop_length
    if audio_len > TEN_MINUTES_SAMPLES:
        target_frames = 2000
        hop_length = max(hop_length, audio_len // target_frames)

    # 計算 STFT（使用原始 sample_rate 保留完整頻率範圍）
    from core.fft import compute_stft
    times, frequencies, spectrogram_db = compute_stft(
        audio, sample_rate, hop_length=hop_length
    )

    # 限制頻率範圍
    freq_mask = frequencies <= fmax
    frequencies = frequencies[freq_mask]
    spectrogram_db = spectrogram_db[freq_mask, :]

    fig, ax = plt.subplots(figsize=figsize)

    # 繪製 Spectrogram
    im = ax.pcolormesh(
        times, frequencies, spectrogram_db,
        shading='gouraud',
        cmap=cmap,
        vmin=np.max(spectrogram_db) - 80,  # 動態範圍 80 dB
        vmax=np.max(spectrogram_db)
    )

    # 設定對數頻率軸
    ax.set_yscale('symlog', linthresh=100)

    # 設定軸標籤
    ax.set_xlabel('時間 (秒)', fontsize=12)
    ax.set_ylabel('頻率 (Hz)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 加入色條
    cbar = fig.colorbar(im, ax=ax, label='幅度 (dB)')

    # 調整布局
    plt.tight_layout()

    # 轉換為 bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    png_bytes = buf.getvalue()
    plt.close(fig)

    return png_bytes


def plot_filtered_spectrum(
    frequencies: np.ndarray,
    magnitudes_original: np.ndarray,
    magnitudes_filtered: np.ndarray,
    cutoff_freq: float,
    title: str = "高頻濾波頻譜對比圖"
) -> bytes:
    """繪製濾波前後對比頻譜圖

    Args:
        frequencies: 頻率陣列 (Hz)
        magnitudes_original: 原始能量陣列 (dB)
        magnitudes_filtered: 濾波後能量陣列 (dB)
        cutoff_freq: 截止頻率 (Hz)
        title: 圖表標題

    Returns:
        bytes: PNG 圖片的 bytes 資料
    """
    fig, ax = plt.subplots(figsize=config.visualization.spectrum_figsize)

    # 繪製原始頻譜
    ax.semilogx(
        frequencies, magnitudes_original,
        'b-', linewidth=0.8, alpha=0.5, label='原始頻譜'
    )

    # 繪製濾波後頻譜
    ax.semilogx(
        frequencies, magnitudes_filtered,
        'r-', linewidth=1.2, label='濾波後頻譜'
    )

    # 標示截止頻率
    ax.axvline(
        x=cutoff_freq, color='g', linestyle='--',
        linewidth=1.5, label=f'截止頻率 ({cutoff_freq} Hz)'
    )

    # 設定軸標籤
    ax.set_xlabel('頻率 (Hz)', fontsize=12)
    ax.set_ylabel('幅度 (dB)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 設定範圍
    if len(frequencies) > 0:
        ax.set_xlim(20, min(20000, frequencies[-1]))

    # 設定網格和圖例
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right')

    # 調整布局
    plt.tight_layout()

    # 轉換為 bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    png_bytes = buf.getvalue()
    plt.close(fig)

    return png_bytes


def plot_band_analysis(
    band_data: dict,
    title: str = "頻帶能量分析"
) -> bytes:
    """繪製頻帶能量分析圖

    Args:
        band_data: 各頻帶分析結果
        title: 圖表標題

    Returns:
        bytes: PNG 圖片的 bytes 資料
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 準備資料
    band_names = []
    energies = []
    statuses = []
    colors = []

    status_colors = {
        'PASS': '#4CAF50',    # 綠色
        'WARNING': '#FF9800', # 橘色
        'FAIL': '#F44336'     # 紅色
    }

    for band_name, data in band_data.items():
        # 轉換頻帶名稱為可讀格式
        display_name = {
            'low_freq': '低頻\n(20-500Hz)',
            'mid_freq': '中頻\n(0.5-2kHz)',
            'mid_high_freq': '中高頻\n(2-6kHz)',
            'high_freq': '高頻\n(6-12kHz)',
            'ultra_high_freq': '超高頻\n(12-20kHz)'
        }.get(band_name, band_name)

        band_names.append(display_name)
        energy = data.get('energy_db', -100)
        # 過濾 -inf 和 NaN 值，設定最小值為 -100 dB
        if not np.isfinite(energy):
            energy = -100.0
        energies.append(energy)
        status = data.get('status', 'PASS')
        statuses.append(status)
        colors.append(status_colors.get(status, '#9E9E9E'))

    # 繪製長條圖
    bars = ax.bar(band_names, energies, color=colors, edgecolor='black')

    # 在長條上方顯示數值
    for bar, energy, status in zip(bars, energies, statuses):
        height = bar.get_height()
        ax.annotate(
            f'{energy:.1f} dB\n({status})',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=9
        )

    # 設定軸標籤
    ax.set_xlabel('頻帶', fontsize=12)
    ax.set_ylabel('能量 (dB)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 設定 Y 軸範圍 (確保有效值)
    valid_energies = [e for e in energies if np.isfinite(e)]
    if valid_energies:
        y_min = min(valid_energies) - 10
        y_max = max(valid_energies) + 15
    else:
        y_min, y_max = -110, 0
    ax.set_ylim(y_min, y_max)

    # 加入網格
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)

    # 調整布局
    plt.tight_layout()

    # 轉換為 bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    png_bytes = buf.getvalue()
    plt.close(fig)

    return png_bytes


def plot_a_weighting_curve(
    sample_rate: int = 48000,
    title: str = "A-Weighting 頻率響應曲線"
) -> bytes:
    """繪製 A-weighting 頻率響應曲線

    Args:
        sample_rate: 取樣率 (Hz)
        title: 圖表標題

    Returns:
        bytes: PNG 圖片的 bytes 資料
    """
    from core.noise_level import get_a_weighting_curve

    frequencies, response_db = get_a_weighting_curve(sample_rate, n_points=2048)

    fig, ax = plt.subplots(figsize=config.visualization.spectrum_figsize)

    # 繪製曲線
    ax.semilogx(frequencies, response_db, 'b-', linewidth=1.5)

    # 設定範圍
    ax.set_xlim(20, 20000)
    ax.set_ylim(-50, 10)

    # 標示重要頻率點
    key_freqs = [20, 100, 500, 1000, 2000, 4000, 10000, 20000]
    ax.set_xticks(key_freqs)
    ax.set_xticklabels([str(f) for f in key_freqs])

    # 加入 0 dB 參考線
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.5)

    # 設定軸標籤
    ax.set_xlabel('頻率 (Hz)', fontsize=12)
    ax.set_ylabel('響應 (dB)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # 設定網格
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    # 調整布局
    plt.tight_layout()

    # 轉換為 bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    png_bytes = buf.getvalue()
    plt.close(fig)

    return png_bytes


def image_to_base64(image_bytes: bytes) -> str:
    """將圖片 bytes 轉換為 base64 字串

    Args:
        image_bytes: PNG 圖片的 bytes 資料

    Returns:
        str: base64 編碼字串
    """
    return base64.b64encode(image_bytes).decode('utf-8')


def save_image(image_bytes: bytes, file_path: str) -> None:
    """儲存圖片至檔案

    Args:
        image_bytes: PNG 圖片的 bytes 資料
        file_path: 儲存路徑
    """
    with open(file_path, 'wb') as f:
        f.write(image_bytes)
