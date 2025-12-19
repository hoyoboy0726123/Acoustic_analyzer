# 聲學測試 AI 分析系統

基於 AI 的筆記型電腦聲學測試分析系統，透過 FFT 頻譜分析與數位濾波技術，實現噪音量測、Discrete Tone 檢測與高頻音隔離分析。

## ✨ 功能特色

- 🎵 **音檔上傳與驗證** - 支援 WAV、MP3、FLAC 格式
- 📊 **FFT 頻譜分析** - 計算各頻率能量分布
- 🔊 **噪音等級計算** - dB(A) 計算，符合 ECMA-74 標準
- 🎯 **Discrete Tone 檢測** - 依據 ECMA-74 Annex D 標準
- ⚡ **高頻音隔離分析** - 電感嘯叫 (Coil Whine) 檢測
- 📈 **頻譜瀑布圖生成** - Spectrogram 視覺化
- 📝 **測試報告自動生成** - 完整分析報告輸出

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
