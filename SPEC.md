# 聲學測試 AI 分析系統 - 軟體需求規格書

> **版本**: 1.0  
> **日期**: 2025-01-15  
> **用途**: 供 Claude Code 開發使用

---

## 1. 專案概述

### 1.1 專案目標

開發一套基於 AI 的筆記型電腦聲學測試分析系統，透過 FFT 頻譜分析與數位濾波技術，實現：
- 噪音量測
- Discrete Tone 檢測
- 高頻音隔離分析（電感嘯叫檢測）

### 1.2 技術堆疊

| 項目 | 技術選型 |
|------|----------|
| 程式語言 | Python 3.10+ |
| Web 框架 | FastAPI (API) / Streamlit (UI) |
| 音訊處理 | librosa, scipy.signal, numpy |
| 濾波器 | scipy.signal.butter, filtfilt |
| 視覺化 | matplotlib, plotly |

---

## 2. 功能需求總覽

| 編號 | 功能名稱 | 優先級 | Phase |
|------|----------|--------|-------|
| AUD-001 | 音檔上傳與格式驗證 | P0 | MVP |
| AUD-002 | FFT 頻譜分析 | P0 | MVP |
| AUD-003 | 噪音等級計算 dB(A) | P0 | MVP |
| AUD-004 | Discrete Tone 檢測 | P0 | MVP |
| AUD-005 | 高頻音隔離檢測 | P0 | MVP |
| AUD-006 | 頻帶分離分析 | P1 | Phase 2 |
| AUD-007 | 頻譜瀑布圖生成 | P1 | Phase 2 |
| AUD-008 | 測試報告自動生成 | P0 | MVP |
| AUD-009 | Web UI 介面 | P0 | MVP |

---

## 3. 詳細功能規格

### 3.1 AUD-001 音檔上傳與格式驗證

**功能描述**: 接收使用者上傳的音訊檔案，驗證格式與品質。

**輸入規格**:
```yaml
file_format: WAV (必須), MP3, FLAC (可選)
sample_rate: 44100 or 48000 Hz
bit_depth: 16-bit or 24-bit
channels: Mono
max_file_size: 50MB
duration: 10-120 seconds
```

**輸出 (JSON)**:
```json
{
  "file_valid": true,
  "sample_rate": 48000,
  "duration": 30.5,
  "channels": 1,
  "bit_depth": 24,
  "error_message": null
}
```

**實作檔案**: `core/audio_loader.py`

---

### 3.2 AUD-002 FFT 頻譜分析

**功能描述**: 使用 FFT 將時域音訊轉換為頻域，計算各頻率能量分布。

**演算法參數**:
```python
n_fft = 4096        # FFT 點數
hop_length = 512    # 幀移動步長
window = 'hann'     # 窗函數
freq_range = (20, 20000)  # Hz
```

**輸出**:
- `frequencies`: List[float] - 頻率陣列 (Hz)
- `magnitudes`: List[float] - 對應能量值 (dB)
- `spectrum_image`: bytes - PNG 格式頻譜圖

**實作檔案**: `core/fft.py`

---

### 3.3 AUD-003 噪音等級計算 dB(A)

**功能描述**: 計算 A 加權聲壓級，符合 ECMA-74 標準。

**計算流程**:
1. 對音訊套用 A-weighting 濾波器
2. 計算 RMS 值
3. 轉換為分貝: `dB = 20 × log₁₀(rms / ref)`, ref = 20 μPa
4. 輸出 Leq、Lmax、Lmin

**輸出 (JSON)**:
```json
{
  "leq_dba": 35.2,
  "lmax_dba": 42.1,
  "lmin_dba": 28.3,
  "l10": 38.5,
  "l90": 30.1
}
```

**實作檔案**: `core/noise_level.py`

---

### 3.4 AUD-004 Discrete Tone 檢測

**功能描述**: 依據 ECMA-74 Annex D 標準檢測突出單頻成分。

**判定標準 (ECMA-74)**:
| 頻率範圍 | 突出量門檻 | 判定 |
|----------|------------|------|
| 89.1 - 282 Hz | > 8 dB | Tone Detected |
| 282 - 893 Hz | > 5 dB | Tone Detected |
| 893 - 11200 Hz | > 3 dB | Tone Detected |

**輸出 (JSON)**:
```json
{
  "tone_detected": true,
  "tones": [
    { "frequency": 4850, "prominence": 6.2, "band": "893-11200Hz" }
  ]
}
```

**實作檔案**: `core/discrete_tone.py`

---

### 3.5 AUD-005 高頻音隔離檢測 ⭐ 重點功能

**功能描述**: 透過數位濾波器將高頻音與低頻風扇噪音分離，獨立分析高頻成分，檢測電感嘯叫 (Coil Whine)、高頻共振等問題。

#### 頻帶定義

| 頻帶 | 頻率範圍 | 常見噪音來源 | 濾波器 | 重要性 |
|------|----------|--------------|--------|--------|
| 低頻 | 20-500 Hz | 風扇轉動、硬碟馬達 | Lowpass | 背景 |
| 中頻 | 500-2k Hz | 風切聲、軸承噪音 | Bandpass | 背景 |
| 中高頻 | 2k-6k Hz | 風扇共振、結構共振 | Bandpass | 注意 |
| 高頻 | 6k-12k Hz | 電感嘯叫 (Coil Whine) | Highpass | **重要** |
| 超高頻 | 12k-20k Hz | 電子雜訊、電源相關 | Highpass | 注意 |

#### 濾波器規格

```python
filter_type = 'butterworth'
filter_order = 5
implementation = 'scipy.signal.butter + scipy.signal.filtfilt'  # 零相位濾波
default_highpass_cutoff = 4000  # Hz
```

#### 高頻異常判定標準

| 檢測項目 | 頻率範圍 | 判定標準 | 說明 |
|----------|----------|----------|------|
| 電感嘯叫 | 6k-12k Hz | 突出量 > 10 dB | GPU/VRM 常見問題 |
| 高頻共振 | 4k-8k Hz | 有明顯峰值 | 風扇/結構共振 |
| 電子雜訊 | 8k-16k Hz | 依規格判定 | 電源相關 |
| 超高頻異常 | 16k-20k Hz | 有明顯峰值需注意 | 年輕人較敏感 |

#### 演算法流程

```python
def analyze_high_frequency(audio_path, cutoff=4000):
    # 1. 讀取音檔
    y, sr = librosa.load(audio_path, sr=48000)
    
    # 2. 設計 Butterworth 高通濾波器
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    b, a = scipy.signal.butter(5, normalized_cutoff, btype='high')
    
    # 3. 零相位濾波
    y_filtered = scipy.signal.filtfilt(b, a, y)
    
    # 4. FFT 分析濾波後訊號
    frequencies, magnitudes = compute_fft(y_filtered, sr)
    
    # 5. 偵測峰值
    peaks = scipy.signal.find_peaks(magnitudes, prominence=6)
    
    # 6. 判定是否異常
    coil_whine = detect_coil_whine(peaks, frequencies, magnitudes)
    
    # 7. 生成報告
    return generate_report(...)
```

#### 輸出規格 (JSON)

```json
{
  "band_analysis": {
    "low_freq": { "range": "20-500Hz", "energy_db": 35.1, "status": "PASS" },
    "mid_freq": { "range": "500-2kHz", "energy_db": 32.4, "status": "PASS" },
    "mid_high_freq": { "range": "2k-6kHz", "energy_db": 28.7, "status": "PASS" },
    "high_freq": { "range": "6k-12kHz", "energy_db": 31.2, "status": "WARNING" },
    "ultra_high_freq": { "range": "12k-20kHz", "energy_db": 18.3, "status": "PASS" }
  },
  "high_freq_peaks": [
    { "frequency": 8420, "magnitude_db": 31.2, "prominence_db": 8.5 }
  ],
  "coil_whine_detected": true,
  "coil_whine_frequency": 8420,
  "coil_whine_prominence": 8.5,
  "possible_cause": "VRM 電感嘯叫",
  "recommendation": "負載變化時複測確認",
  "overall_status": "WARNING",
  "filtered_spectrum_image": "<base64_png>"
}
```

**實作檔案**: `core/filters.py`, `core/high_freq_detector.py`

---

### 3.6 AUD-006 頻帶分離分析

**功能描述**: 依據 IEC 61260-1 標準，使用 FFT Synthesis (STFT Avg) 方法進行 1/3 倍頻程分析 (20Hz-20kHz)。確保與 FFT 頻譜圖視覺一致性。

**輸出**: 各頻帶的 dB 值、佔比百分比、主要噪音來源判斷。

**實作檔案**: `core/band_analyzer.py`

---

### 3.7 AUD-007 頻譜瀑布圖生成

**功能描述**: 生成 Spectrogram，顯示頻率能量隨時間變化。

**參數**:
```python
cmap = 'viridis'
y_axis = 'log'
fmax = 20000  # Hz
image_size = (1200, 600)  # pixels
```

**實作檔案**: `utils/visualization.py`

---

### 3.8 AUD-008 測試報告自動生成

**功能描述**: 產生包含所有分析結果的測試報告。

**報告區塊**:
- 基本資訊 (機型、日期、測試情境)
- 全頻段分析 (總噪音 dB(A)、Pass/Fail)
- 分頻段分析表格
- Discrete Tone 結果
- 高頻異常詳情
- 圖表 (頻譜圖、高頻濾波圖、Spectrogram)

**報告格式範例**:
```
┌────────────────────────────────────────────────┐
│  聲學測試報告 - 頻帶分離分析                   │
│  機型: ASUS Zenbook 14                         │
│  測試情境: Stress Mode                         │
├────────────────────────────────────────────────┤
│                                                │
│  【全頻段分析】                                │
│    總噪音: 38.2 dB(A)  ✅ PASS                 │
│                                                │
│  【分頻段分析】                                │
│  ┌──────────────────────────────────────────┐ │
│  │ 頻段        │ 能量    │ 狀態   │ 來源   │ │
│  ├──────────────────────────────────────────┤ │
│  │ 低頻 <500Hz │ 35.1 dB │ ✅     │ 風扇   │ │
│  │ 中頻 0.5-2k │ 32.4 dB │ ✅     │ 風切   │ │
│  │ 中高 2k-6k  │ 28.7 dB │ ✅     │ -      │ │
│  │ 高頻 6k-12k │ 31.2 dB │ ⚠️     │ 電感?  │ │
│  │ 超高 >12kHz │ 18.3 dB │ ✅     │ -      │ │
│  └──────────────────────────────────────────┘ │
│                                                │
│  【高頻異常詳情】                              │
│    - 偵測到 8,420 Hz 突出峰值                  │
│    - 突出量: +8.5 dB (建議 < 6 dB)             │
│    - 可能原因: VRM 電感嘯叫                    │
│    - 建議: 負載變化時複測確認                  │
│                                                │
│  [全頻譜圖] [高頻濾波圖] [播放高頻音] [報告]   │
└────────────────────────────────────────────────┘
```

**實作檔案**: `utils/report.py`

---

## 4. API 設計規格

### 4.1 RESTful API 端點

| Method | Endpoint | 說明 |
|--------|----------|------|
| POST | `/api/v1/audio/upload` | 上傳音檔 |
| POST | `/api/v1/audio/analyze` | 執行完整音頻分析 |
| POST | `/api/v1/audio/analyze/spectrum` | FFT 頻譜分析 |
| POST | `/api/v1/audio/analyze/noise-level` | 噪音等級計算 |
| POST | `/api/v1/audio/analyze/discrete-tone` | Discrete Tone 檢測 |
| POST | `/api/v1/audio/analyze/high-freq` | 高頻音隔離分析 |
| POST | `/api/v1/audio/analyze/spectrogram` | 頻譜瀑布圖生成 |
| GET | `/api/v1/report/{task_id}` | 取得分析報告 |

### 4.2 高頻音分析 API 範例

**POST /api/v1/audio/analyze/high-freq**

Request (multipart/form-data):
```
file: audio.wav
filter_cutoff: 4000  // Hz
detect_coil_whine: true
```

Response (application/json):
```json
{
  "task_id": "abc123",
  "status": "completed",
  "band_analysis": { ... },
  "high_freq_peaks": [{ "freq": 8420, "magnitude_db": 31.2, "prominence_db": 8.5 }],
  "coil_whine_detected": true,
  "overall_status": "WARNING"
}
```

---

## 5. 專案目錄結構

```
acoustic-ai-analyzer/
├── app/
│   ├── main.py                    # FastAPI 入口
│   ├── config.py                  # 設定檔
│   ├── routers/
│   │   └── audio.py               # 音頻分析 API
│   └── schemas/
│       └── audio.py               # Pydantic 資料模型
├── core/
│   ├── audio_loader.py            # 音檔載入與驗證
│   ├── fft.py                     # FFT 頻譜分析
│   ├── filters.py                 # 濾波器 (高頻隔離)
│   ├── discrete_tone.py           # Discrete Tone 檢測
│   ├── noise_level.py             # 噪音等級計算
│   ├── high_freq_detector.py      # 高頻異常檢測
│   └── band_analyzer.py           # 頻帶分離分析
├── utils/
│   ├── visualization.py           # 頻譜圖、Spectrogram
│   └── report.py                  # 報告生成
├── ui/
│   └── streamlit_app.py           # Streamlit Web UI
├── tests/
│   ├── test_fft.py
│   ├── test_filters.py
│   └── test_high_freq.py
├── sample_audio/                  # 測試用音檔
├── requirements.txt
├── SPEC.md                        # 本文件
└── README.md
```

---

## 6. 開發任務清單 (Tasks)

### Phase 1 - MVP (Week 1-2)

- [ ] **Task 1**: 專案初始化
  - 建立目錄結構
  - 建立 requirements.txt
  - 建立 config.py

- [ ] **Task 2**: 實作 AUD-001 音檔載入
  - 檔案: `core/audio_loader.py`
  - 功能: 讀取 WAV、驗證格式、回傳 metadata

- [ ] **Task 3**: 實作 AUD-002 FFT 分析
  - 檔案: `core/fft.py`
  - 功能: FFT 計算、頻譜資料輸出

- [ ] **Task 4**: 實作 AUD-003 噪音等級計算
  - 檔案: `core/noise_level.py`
  - 功能: A-weighting、dB(A) 計算

- [ ] **Task 5**: 實作 AUD-005 高頻濾波器
  - 檔案: `core/filters.py`
  - 功能: Butterworth highpass filter

- [ ] **Task 6**: 實作 AUD-005 高頻檢測器
  - 檔案: `core/high_freq_detector.py`
  - 功能: 峰值偵測、電感嘯叫判定

- [ ] **Task 7**: 實作頻譜圖視覺化
  - 檔案: `utils/visualization.py`
  - 功能: 全頻譜圖、濾波後頻譜圖

- [ ] **Task 8**: 實作 Streamlit UI
  - 檔案: `ui/streamlit_app.py`
  - 功能: 檔案上傳、分析結果顯示

### Phase 2 - 完整功能 (Week 3-4)

- [ ] **Task 9**: 實作 AUD-004 Discrete Tone
  - 檔案: `core/discrete_tone.py`

- [ ] **Task 10**: 實作 AUD-006 頻帶分離
  - 檔案: `core/band_analyzer.py`

- [ ] **Task 11**: 實作 AUD-007 Spectrogram
  - 檔案: `utils/visualization.py` (擴充)

- [ ] **Task 12**: 實作 AUD-008 報告生成
  - 檔案: `utils/report.py`

- [ ] **Task 13**: 實作 FastAPI 路由
  - 檔案: `app/routers/audio.py`, `app/main.py`

- [ ] **Task 14**: 撰寫單元測試
  - 檔案: `tests/`

---

## 7. 驗收標準

| 項目 | 標準 |
|------|------|
| 音檔分析時間 | < 5 秒 (30秒音檔) |
| 高頻音檢測準確率 | > 90% |
| Discrete Tone 準確率 | > 95% |
| API 回應時間 | < 100ms (不含分析) |
| 支援音檔格式 | WAV (必須) |

---

## 8. Dependencies (requirements.txt)

```
numpy>=1.24.0
scipy>=1.10.0
librosa>=0.10.0
matplotlib>=3.7.0
plotly>=5.14.0
fastapi>=0.100.0
uvicorn>=0.22.0
streamlit>=1.24.0
python-multipart>=0.0.6
pydantic>=2.0.0
```

---

## 附錄: Claude Code 使用指令

### 開始開發

```bash
# 在專案目錄執行
claude

# 然後輸入
請閱讀 SPEC.md，從 Task 1 開始按順序開發這個聲學測試 AI 系統
```

### 單獨執行某個 Task

```
請執行 Task 5: 實作高頻濾波器 (core/filters.py)
```

### 檢視進度

```
請檢視目前的開發進度，哪些 Task 已完成？
```