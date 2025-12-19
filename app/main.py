# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - FastAPI 主入口

此檔案是 FastAPI 應用程式的主要入口點，負責：
- 初始化 FastAPI 應用程式
- 載入路由
- 設定 CORS 與中間件
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import config

# 建立 FastAPI 應用程式實例
app = FastAPI(
    title="聲學測試 AI 分析系統",
    description="基於 AI 的筆記型電腦聲學測試分析系統，提供 FFT 頻譜分析、噪音量測、Discrete Tone 檢測與高頻音隔離分析功能。",
    version="1.0.0",
    docs_url=f"{config.api_prefix}/docs",
    redoc_url=f"{config.api_prefix}/redoc",
    openapi_url=f"{config.api_prefix}/openapi.json"
)

# 設定 CORS (跨來源資源共享)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境應限制來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """根路徑端點，回傳系統資訊"""
    return {
        "name": "聲學測試 AI 分析系統",
        "version": "1.0.0",
        "status": "running",
        "api_docs": f"{config.api_prefix}/docs"
    }


@app.get("/health")
async def health_check():
    """健康檢查端點"""
    return {"status": "healthy"}


# TODO: 在 Task 13 時加入路由
# from app.routers import audio
# app.include_router(audio.router, prefix=config.api_prefix)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=config.debug
    )
