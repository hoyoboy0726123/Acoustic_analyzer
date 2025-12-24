# -*- coding: utf-8 -*-
"""
同步音訊播放器元件

提供 HTML5 Audio 播放器與 Spectrogram 進度線同步顯示功能
使用 Plotly.js 實現互動式圖表（縮放、平移）
與 interactive_plots.py 中的 create_spectrogram_chart 完全一致
"""

import streamlit as st
import streamlit.components.v1 as components
import base64
import io
import soundfile as sf
import numpy as np
import json


def create_audio_player_with_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    title: str = "🎵 同步音訊播放器",
    fmax: int = 20000,
    n_fft: int = 2048,
    hop_length: int = 512,
    calibration_offset: float = 0.0,
    use_a_weighting: bool = True,
    spl_offset: float = 0.0
) -> None:
    """建立帶有互動式 Spectrogram 和進度線的音訊播放器
    
    使用與 create_spectrogram_chart 完全相同的參數設定，
    確保播放器與分析區域的 Spectrogram 完全一致。
    
    使用 Plotly.js 實現：
    - 滑鼠滾輪縮放
    - 拖曳平移
    - 雙擊重置
    - 播放進度線同步
    - 點擊跳轉播放位置
    
    Args:
        audio: 音訊資料
        sample_rate: 取樣率
        title: 標題
        fmax: 最大顯示頻率 (Hz)，預設 20000
        n_fft: FFT 視窗大小，預設 2048
        hop_length: 跳躍長度，預設 512
        calibration_offset: 麥克風校準偏移 (dB)
        use_a_weighting: 是否套用 A-weighting
        spl_offset: dB SPL 模式偏移 (dB)
    """
    from scipy.signal import spectrogram as scipy_spectrogram
    from core.noise_level import apply_a_weighting
    
    # 將音訊轉換為 base64 (使用原始音訊，不影響播放)
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio, sample_rate, format='WAV')
    audio_buffer.seek(0)
    audio_base64 = base64.b64encode(audio_buffer.read()).decode()
    
    # === 長音訊優化（超過 10 分鐘）===
    TEN_MINUTES_SAMPLES = sample_rate * 600
    audio_len = len(audio)
    MAX_TIME_BINS = 2000
    
    # 對長音訊增加 hop_length
    effective_hop = hop_length
    if audio_len > TEN_MINUTES_SAMPLES:
        target_frames = MAX_TIME_BINS
        effective_hop = max(hop_length, audio_len // target_frames)
    
    # 如果啟用 A-weighting，對音訊進行加權
    if use_a_weighting:
        audio_for_spec = apply_a_weighting(audio, sample_rate)
    else:
        audio_for_spec = audio
    
    # 計算 Spectrogram（使用原始 sample_rate）
    frequencies, times, Sxx = scipy_spectrogram(
        audio_for_spec, fs=sample_rate,
        nperseg=n_fft, noverlap=max(0, n_fft - effective_hop)
    )
    
    # 轉換為 dB 並套用校準偏移（麥克風校準 + SPL 偏移）
    total_offset = calibration_offset + spl_offset
    Sxx_db = 10 * np.log10(Sxx + 1e-10) + total_offset
    
    # 限制頻率範圍
    freq_mask = frequencies <= min(fmax, sample_rate / 2)
    frequencies = frequencies[freq_mask]
    Sxx_db = Sxx_db[freq_mask, :]
    
    # 動態單位標籤
    unit_label = "dB(A)" if use_a_weighting else "dB"
    if spl_offset > 0:
        unit_label += " SPL"
    
    # 音訊長度
    duration = len(audio) / sample_rate
    
    # 計算初始頻率範圍 (對數軸)
    freq_min = 20
    freq_max = min(fmax, sample_rate / 2)
    
    # 智能降採樣 - 限制最大解析度以避免資料過大
    # 但維持足夠的解析度以確保視覺質量
    max_time_points = 1000  # 時間軸最大點數
    max_freq_points = 400   # 頻率軸最大點數
    
    time_step = max(1, len(times) // max_time_points)
    freq_step = max(1, len(frequencies) // max_freq_points)
    
    times_sub = times[::time_step]
    freq_sub = frequencies[::freq_step]
    Sxx_sub = Sxx_db[::freq_step, ::time_step]
    
    # 準備 Plotly 資料
    times_json = json.dumps(times_sub.tolist())
    freq_json = json.dumps(freq_sub.tolist())
    z_json = json.dumps(Sxx_sub.tolist())
    
    # 生成包含 Plotly.js 的 HTML
    html_code = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            .player-container {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                padding: 12px;
                background: white;
                border: 1px solid #e0e0e0;
                border-radius: 12px;
                color: #333;
            }}
            .title {{
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 8px;
                color: #333;
            }}
            .hint {{
                font-size: 11px;
                color: #666;
                margin-bottom: 8px;
            }}
            #plotlyChart {{
                width: 100%;
                height: 350px;
                border-radius: 8px;
                overflow: hidden;
            }}
            .audio-controls {{
                margin-top: 10px;
            }}
            audio {{
                width: 100%;
                height: 40px;
                border-radius: 8px;
            }}
            .time-info {{
                display: flex;
                justify-content: space-between;
                font-size: 11px;
                color: #888;
                margin-top: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="player-container">
            <div class="title">{title}</div>
            <div class="hint">💡 滾輪縮放 | 拖曳平移 | 雙擊重置 | 點擊圖表跳轉播放位置</div>
            
            <div id="plotlyChart"></div>
            
            <div class="audio-controls">
                <audio id="audioPlayer" controls>
                    <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                </audio>
            </div>
            
            <div class="time-info">
                <span id="currentTime">0:00.0</span>
                <span>總長: {duration:.1f} 秒</span>
            </div>
        </div>
        
        <script>
            // Spectrogram 資料
            const times = {times_json};
            const frequencies = {freq_json};
            const zData = {z_json};
            const duration = {duration};
            
            // 建立 Plotly 圖表
            const heatmapTrace = {{
                x: times,
                y: frequencies,
                z: zData,
                type: 'heatmap',
                colorscale: 'Viridis',
                colorbar: {{
                    title: {{ text: '{unit_label}', side: 'right' }},
                    thickness: 15,
                    len: 0.9
                }},
                hovertemplate: '時間: %{{x:.2f}}s<br>頻率: %{{y:.0f}} Hz<br>幅度: %{{z:.1f}} {unit_label}<extra></extra>'
            }};
            
            // 進度線 (初始位置)
            const progressLine = {{
                type: 'line',
                x0: 0, x1: 0,
                y0: 20,
                y1: {freq_max},
                line: {{ color: 'red', width: 3 }}
            }};
            
            const layout = {{
                margin: {{ l: 60, r: 100, t: 10, b: 60 }},
                paper_bgcolor: 'white',
                plot_bgcolor: 'white',
                xaxis: {{
                    title: '時間 (秒)',
                    showgrid: true,
                    gridcolor: 'rgba(128, 128, 128, 0.3)',
                    range: [0, duration]
                }},
                yaxis: {{
                    title: '頻率 (Hz)',
                    showgrid: true,
                    gridcolor: 'rgba(128, 128, 128, 0.3)',
                    type: 'log',
                    range: [Math.log10(20), Math.log10({freq_max})]
                }},
                shapes: [progressLine],
                dragmode: 'zoom'
            }};
            
            const config = {{
                responsive: true,
                scrollZoom: true,
                displayModeBar: true,
                modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                displaylogo: false
            }};
            
            Plotly.newPlot('plotlyChart', [heatmapTrace], layout, config);
            
            // 音訊播放器
            const audio = document.getElementById('audioPlayer');
            const currentTimeDisplay = document.getElementById('currentTime');
            const plotDiv = document.getElementById('plotlyChart');
            
            // 更新進度線
            function updateProgressLine(time) {{
                Plotly.relayout('plotlyChart', {{
                    'shapes[0].x0': time,
                    'shapes[0].x1': time
                }});
                
                const mins = Math.floor(time / 60);
                const secs = (time % 60).toFixed(1);
                currentTimeDisplay.textContent = mins + ':' + (secs < 10 ? '0' : '') + secs;
            }}
            
            // 播放時間更新
            audio.addEventListener('timeupdate', function() {{
                updateProgressLine(audio.currentTime);
            }});
            
            // 點擊圖表跳轉播放位置
            plotDiv.on('plotly_click', function(data) {{
                if (data.points && data.points[0]) {{
                    const clickedTime = data.points[0].x;
                    audio.currentTime = clickedTime;
                    updateProgressLine(clickedTime);
                }}
            }});
            
            // 監聽圖表範圍變化，保持進度線可見
            plotDiv.on('plotly_relayout', function(eventdata) {{
                // 範圍變化時不需特殊處理
            }});
        </script>
    </body>
    </html>
    '''
    
    components.html(html_code, height=500)



def create_simple_audio_player(
    audio: np.ndarray,
    sample_rate: int,
    label: str = "音訊"
) -> None:
    """建立簡單的音訊播放器
    
    Args:
        audio: 音訊資料
        sample_rate: 取樣率
        label: 標籤
    """
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio, sample_rate, format='WAV')
    audio_buffer.seek(0)
    
    st.caption(f"▶️ {label}")
    st.audio(audio_buffer, format='audio/wav')
