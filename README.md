# 聲學測試分析系統

本系統專為筆記型電腦聲學測試設計，透過 FFT 頻譜分析與數位濾波技術，實現噪音量測、Discrete Tone 檢測與高頻音隔離分析。

## 📘 聲學分析參數指南 (Acoustic Parameter Guide)

為符合業界標準 (如 ISO, ECMA-74) 及 HEAD acoustics ArtemiS 分析習慣，建議設定如下：

### 1. 分析模式 (Analysis Mode)
*   **FFT Average (推薦常用)**：計算時間內的平均能量，適合穩態噪音（如風扇聲）。
*   **FFT Peak Hold**：保留各頻率出現過的最大值，適合抓瞬時異音。
*   **PSD**：功率頻譜密度，用於隨機訊號分析。

### 2. 窗函數 (Window Function)
*   **Hann (推薦常用)**：通用標準，頻率解析度與洩漏抑制的最佳平衡。
*   **Blackman**：旁瓣抑制極佳，但主瓣較寬。
*   **Flat Top**：幅度最精準，專用於校準器讀值驗證。

### 3. FFT 點數 (Resolution)
*   **8192 (推薦常用)**：解析度約 5.9 Hz (以 48kHz 取樣率為例)，平衡了時間與頻率的精細度。

---

## 🎤 麥克風校準指南 (Microphone Calibration Guide)

### 為什麼需要校準？

電腦麥克風（無論是內建麥克風、USB 麥克風或外接錄音介面）錄製的音訊是**相對數位值**，並非真實的聲壓級 (dB SPL)。不同的麥克風、音效卡增益設定、作業系統音量都會影響錄製結果。

**校準的目的**：
*   將數位訊號轉換為真實的物理單位 (dB SPL)。
*   讓不同設備、不同時間的測量結果具有**可比較性**。
*   符合聲學測試標準 (如 ISO, ECMA-74) 的量測要求。

### 校準偏移 (Calibration Offset) 的意義

`校準偏移 (dB)` 是一個加法常數，用來補償系統靈敏度差異：

```
真實 dB SPL = 系統顯示值 + 校準偏移
```

例如：
*   校準器發出 **94 dB SPL** 的標準聲音。
*   您的系統在未校準時顯示 **Leq = 70 dB**。
*   校準偏移 = 94 - 70 = **+24 dB**。

### 校準步驟 (使用聲學校準器)

這是最精確的方法，需要一個標準聲學校準器（如 94 dB / 114 dB / 1kHz）。

1.  **準備校準器**：將校準器放置於麥克風頭上，確保密封良好。
2.  **錄製校準音**：錄製約 5-10 秒的校準音。
3.  **上傳至系統**：將錄製的 WAV 檔案上傳至本分析系統。
4.  **讀取 Leq**：觀察系統顯示的 `Leq` 值（未校準）。
5.  **計算偏移**：`校準偏移 = 校準器標稱值 - 系統顯示值`
    *   例如：校準器 94 dB，系統顯示 70 dB → 偏移 = +24 dB。
6.  **輸入偏移**：在側邊欄的「麥克風校準」區塊輸入此值。

### 替代校準方法 (無校準器)

如果沒有專業校準器，可使用以下替代方案（精度較低）：

#### 方法 A：使用手機 App 對比
1.  使用經過校準的手機 App（如 NIOSH SLM、Decibel X Pro）測量環境噪音。
2.  同時用您的麥克風錄製相同環境。
3.  比較 App 顯示的 Leq 與本系統顯示的 Leq，差值即為校準偏移。
4.  **注意**：手機 App 本身也需要校準，此方法誤差較大。

#### 方法 B：使用 HEAD acoustics 或專業儀器對比
1.  使用專業聲學分析儀（如 HEAD ArtemiS, B&K Pulse）測量待測物。
2.  使用相同麥克風位置，用您的錄音設備錄製相同音源。
3.  比較兩者的 Leq 或 FFT 幅度，差值即為校準偏移。
4.  **這是工廠環境中最實用的方法**。

### Spectrogram dB SPL 偏移

除了 Leq 校準外，本系統還提供 **Spectrogram dB SPL 偏移** 設定。

**意義**：將 Spectrogram 的相對 dB 色彩軸轉換為絕對 dB SPL，方便與 HEAD acoustics 等專業軟體的 Spectrogram 直接對比。

**取得方式**：
1.  在 HEAD acoustics 中開啟同一音檔的 Spectrogram。
2.  觀察某個已知頻率（如 1kHz）的色彩值（例如顯示 45 dB SPL）。
3.  在本系統中，觀察相同位置的色彩值（例如顯示 -55 dB）。
4.  計算偏移：45 - (-55) = **+100 dB**。
5.  輸入此偏移值即可對齊兩個系統的 Spectrogram 顯示。

### 校準注意事項

- 🔒 **固定錄音參數**：校準後，請勿更改麥克風增益、作業系統音量或錄音軟體設定，否則需重新校準。
- 📏 **定期驗證**：建議每週或每月使用校準器驗證一次。
- 🌡️ **環境因素**：極端溫度和濕度可能影響麥克風靈敏度。
- 📝 **記錄校準值**：建議將校準日期、設備序號、偏移值記錄下來，供日後追溯。

---

## ✨ 功能特色

- 🎵 **音檔上傳與驗證** - 支援 WAV、MP3、FLAC 格式
- 📊 **FFT 頻譜分析** - 計算各頻率能量分布
- 🔊 **噪音等級計算** - dB(A) 計算，符合 ECMA-74 標準
- 🎯 **Discrete Tone 檢測** - 依據 ECMA-74 Annex D 標準
- ⚡ **高頻音隔離分析** - 電感嘯叫 (Coil Whine) 檢測
- 📈 **頻譜瀑布圖生成** - Spectrogram 視覺化
- 📝 **測試報告自動生成** - 完整分析報告輸出 (PDF/Excel)

## 🛠️ 技術堆疊

| 項目 | 技術選型 |
|------|----------|
| 程式語言 | Python 3.10+ |
| Web 框架 | FastAPI (API) / Streamlit (UI) |
| 音訊處理 | librosa, scipy.signal, numpy |
| 濾波器 | scipy.signal.butter, filtfilt |
| 視覺化 | matplotlib, plotly |

## 📦 安裝

```bash
# 安裝依賴套件
pip install -r requirements.txt
```

## 🚀 快速開始

### 方式一：啟動 FastAPI 後端

```bash
python -m app.main
# 或使用 uvicorn
uvicorn app.main:app --reload
```

API 文件: http://localhost:8000/api/v1/docs

### 方式二：啟動 Streamlit UI

```bash
streamlit run ui/streamlit_app.py
```

## 📁 專案結構

```
acoustic-ai-analyzer/
├── app/                    # FastAPI 應用程式
│   ├── main.py            # 入口點
│   ├── config.py          # 設定檔
│   ├── routers/           # API 路由
│   └── schemas/           # 資料模型
├── core/                   # 核心分析模組
│   ├── audio_loader.py    # 音檔載入
│   ├── fft.py             # FFT 分析
│   ├── filters.py         # 濾波器
│   ├── discrete_tone.py   # Discrete Tone
│   ├── noise_level.py     # 噪音等級
│   ├── high_freq_detector.py  # 高頻檢測
│   └── band_analyzer.py   # 頻帶分析
├── utils/                  # 工具模組
│   ├── visualization.py   # 視覺化
│   └── report.py          # 報告生成
├── ui/                     # Web UI
│   └── streamlit_app.py   # Streamlit 介面
├── tests/                  # 單元測試
├── sample_audio/           # 測試音檔
├── requirements.txt        # 依賴套件
├── SPEC.md                # 規格書
└── README.md              # 本文件
```

## 📋 開發進度

- [x] Task 1: 專案初始化
- [x] Task 2: 音檔載入與驗證
- [x] Task 3: FFT 頻譜分析
- [x] Task 4: 噪音等級計算
- [x] Task 5: 高頻濾波器
- [x] Task 6: 高頻檢測器
- [x] Task 7: 頻譜圖視覺化
- [x] Task 8: Streamlit UI
- [x] Task 9: Discrete Tone 檢測
- [x] Task 10: 頻帶分離分析
- [x] Task 11: Spectrogram (已在視覺化模組中)
- [x] Task 12: 報告生成
- [x] Task 13: FastAPI 路由
- [ ] Task 14: 單元測試

## 📄 授權

MIT License

## 📞 聯絡

如有問題，請開啟 Issue。
