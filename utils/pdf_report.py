# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - PDF 報告生成模組

功能:
- 生成專業的 PDF 測試報告
- 嵌入 Plotly 圖表截圖
- 支援中文顯示
"""

import io
import os
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

# PDF 生成
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    Image, PageBreak, HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Plotly 圖表導出
import plotly.graph_objects as go

# 導入分析模組
from core.noise_level import calculate_noise_level
from core.band_analyzer import compute_octave_bands
from core.discrete_tone import detect_discrete_tones
from core.sop_analyzer import analyze_idle_mode, analyze_ue_mode, analyze_workload_mode
from core.fft import compute_average_spectrum, apply_a_weighting
from utils.interactive_plots import create_octave_band_chart


def register_chinese_font():
    """註冊中文字型
    
    嘗試使用系統字型，若失敗則使用預設字型
    """
    chinese_fonts = [
        ("C:/Windows/Fonts/msjh.ttc", "Microsoft JhengHei"),
        ("C:/Windows/Fonts/msyh.ttc", "Microsoft YaHei"),
        ("C:/Windows/Fonts/simsun.ttc", "SimSun"),
    ]
    
    for font_path, font_name in chinese_fonts:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont('ChineseFont', font_path))
                return 'ChineseFont'
            except:
                continue
    
    # 如果沒有中文字型，使用預設
    return 'Helvetica'


def export_plotly_to_image(fig: go.Figure, width: int = 800, height: int = 400) -> Optional[bytes]:
    """將 Plotly 圖表導出為 PNG 圖片
    
    Args:
        fig: Plotly Figure 物件
        width: 圖片寬度
        height: 圖片高度
        
    Returns:
        bytes: PNG 圖片二進位數據，失敗時返回 None
    """
    try:
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception as e:
        # kaleido 在 Windows 上可能有超時問題，忽略並繼續
        print(f"Warning: Failed to export chart: {e}")
        return None


def create_styles(font_name: str) -> Dict[str, ParagraphStyle]:
    """創建 PDF 樣式
    
    Args:
        font_name: 字型名稱
        
    Returns:
        Dict: 各種樣式
    """
    styles = getSampleStyleSheet()
    
    # 標題樣式
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=24,
        textColor=colors.HexColor('#1a1a2e'),
        spaceAfter=30,
        alignment=1  # 置中
    )
    
    # 副標題樣式
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Heading2'],
        fontName=font_name,
        fontSize=16,
        textColor=colors.HexColor('#16213e'),
        spaceBefore=20,
        spaceAfter=10
    )
    
    # 正文樣式
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=11,
        textColor=colors.HexColor('#333333'),
        spaceAfter=8,
        leading=16
    )
    
    # PASS 樣式
    pass_style = ParagraphStyle(
        'Pass',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=36,
        textColor=colors.HexColor('#27ae60'),
        alignment=1,
        spaceBefore=20,
        spaceAfter=20
    )
    
    # FAIL 樣式
    fail_style = ParagraphStyle(
        'Fail',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=36,
        textColor=colors.HexColor('#e74c3c'),
        alignment=1,
        spaceBefore=20,
        spaceAfter=20
    )
    
    # 表格標題樣式
    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        textColor=colors.white,
        alignment=1
    )
    
    return {
        'title': title_style,
        'subtitle': subtitle_style,
        'body': body_style,
        'pass': pass_style,
        'fail': fail_style,
        'table_header': table_header_style
    }


def generate_pdf_report(
    audio_data: np.ndarray,
    sample_rate: int,
    filename: str = "audio.wav",
    sop_params: dict = None,
    analyze_discrete_tone_flag: bool = True,
    calibration_offset: float = 0.0,
    leq_settings: dict = None
) -> Tuple[Optional[bytes], Optional[str]]:
    """生成 PDF 測試報告
    
    Args:
        audio_data: 音訊數據
        sample_rate: 取樣率
        filename: 原始檔名
        sop_params: SOP 分析參數
        analyze_discrete_tone_flag: 是否分析 Discrete Tone
        calibration_offset: 校準偏移量 (dB)
        
    Returns:
        Tuple[bytes, str]: (PDF 檔案二進位數據, 錯誤訊息/None)
    """
    try:
        # 註冊中文字型
        font_name = register_chinese_font()
        styles = create_styles(font_name)
        
        # 建立 PDF 緩衝區
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        # 報告內容
        story = []
        
        # ===== 1. 封面頁 =====
        story.append(Spacer(1, 3*cm))
        story.append(Paragraph("聲學測試分析報告", styles['title']))
        story.append(Paragraph("Acoustic Analysis Report", styles['subtitle']))
        story.append(Spacer(1, 2*cm))
        
        # 計算全域噪音指標
        noise_metrics = calculate_noise_level(audio_data, sample_rate)
        duration = len(audio_data) / sample_rate
        
        # SOP 分析（支援多模式）
        sop_results = {}  # 儲存各模式結果
        overall_pass = True
        
        if sop_params:
            modes = sop_params.get('modes', [sop_params.get('mode', 'IDLE')])
            
            for mode in modes:
                if mode == 'IDLE':
                    spec = sop_params.get('idle_spec', 20.0) - calibration_offset
                    result = analyze_idle_mode(audio_data, sample_rate, spec)
                    sop_results['IDLE'] = result
                    if not result.get('is_pass', True):
                        overall_pass = False
                elif mode == 'UE':
                    result = analyze_ue_mode(audio_data, sample_rate)
                    sop_results['UE'] = result
                elif mode == 'Workload':
                    spec_fail = sop_params.get('work_spec_fail', 22.0) - calibration_offset
                    spec_max = sop_params.get('work_spec_max', 28.0) - calibration_offset
                    result = analyze_workload_mode(audio_data, sample_rate, spec_fail, spec_max)
                    sop_results['Workload'] = result
                    if not result.get('is_pass', True):
                        overall_pass = False
        
        # PASS/FAIL 狀態
        if sop_params and sop_results:
            if overall_pass:
                story.append(Paragraph("✓ PASS", styles['pass']))
            else:
                story.append(Paragraph("✗ FAIL", styles['fail']))
        
        story.append(Spacer(1, 1*cm))
        
        # Discrete Tone 檢測結果（用於第一頁顯示）
        discrete_tone_result = None
        if analyze_discrete_tone_flag:
            from core.discrete_tone import detect_discrete_tones
            discrete_tone_result = detect_discrete_tones(audio_data, sample_rate)
        
        # 基本資訊表格
        info_data = [
            ["檔案名稱", filename],
            ["分析日期", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            ["錄音時長", f"{duration:.2f} 秒"],
            ["取樣率", f"{sample_rate} Hz"],
        ]
        
        # Leq 判定結果
        if leq_settings and leq_settings.get('spec'):
            leq_spec = leq_settings['spec']
            leq_tag = leq_settings.get('tag', 'General')
            # 這裡需要計算一次 Leq (因為我們在下面才計算 noise_metrics)
            # 為了效率，先計算基本的 Leq
            if 'noise_metrics' not in locals():
                noise_metrics = calculate_noise_level(audio_data, sample_rate)
            
            leq_val = noise_metrics['leq_dba'] + calibration_offset
            leq_fail = leq_val > leq_spec
            leq_status = "FAIL" if leq_fail else "PASS"
            info_data.append([f"{leq_tag}", leq_status])
        
        # 添加 Discrete Tone 結果（如果有分析）
        if analyze_discrete_tone_flag and discrete_tone_result:
            tone_detected = discrete_tone_result.get('tone_detected', False)
            tone_status = "FAIL" if tone_detected else "PASS"
            info_data.append(["Discrete Tone", tone_status])
        
        # 添加 SOW 結果（如果有分析）- 支援多模式
        if sop_params and sop_results:
            for mode_name, mode_result in sop_results.items():
                sow_pass = mode_result.get('is_pass', True)
                sow_status = "PASS" if sow_pass else "FAIL"
                info_data.append([f"SOW ({mode_name})", sow_status])
        
        info_table = Table(info_data, colWidths=[5*cm, 10*cm])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8e8e8')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#333333')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
        ]))
        story.append(info_table)
        
        story.append(PageBreak())
        
        # ===== 2. 噪音指標 =====
        story.append(Paragraph("噪音指標摘要", styles['subtitle']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
        story.append(Spacer(1, 0.5*cm))
        
        # 套用校準偏移
        leq = noise_metrics['leq_dba'] + calibration_offset
        lmax = noise_metrics['lmax_dba'] + calibration_offset
        lmin = noise_metrics['lmin_dba'] + calibration_offset
        l10 = noise_metrics['l10'] + calibration_offset
        l90 = noise_metrics['l90'] + calibration_offset
        
        metrics_data = [
            ["指標", "數值", "說明"],
            [Paragraph("<b>Leq</b>", styles['body']), Paragraph(f"<b><font color='#2980b9' size='14'>{leq:.1f} dB(A)</font></b>", styles['body']), "等效連續音壓級"],
            ["Lmax", f"{lmax:.1f} dB(A)", "最大音壓級"],
            ["Lmin", f"{lmin:.1f} dB(A)", "最小音壓級"],
            ["L10", f"{l10:.1f} dB(A)", "超過 10% 時間的音壓級"],
            ["L90", f"{l90:.1f} dB(A)", "超過 90% 時間的音壓級（背景噪音）"],
        ]
        
        metrics_table = Table(metrics_data, colWidths=[4*cm, 5*cm, 8*cm])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
            
            # Leq 行的特殊樣式 (背景色高亮)
            ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor('#eaf2f8')),
            ('TEXTCOLOR', (0, 1), (-1, 1), colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (2, 1), (2, -1), 'LEFT'), # Keep original alignment for description column
            ('FONTSIZE', (0, 1), (-1, -1), 10), # Apply default font size to body rows
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8), # Apply default padding to body rows
            ('TOPPADDING', (0, 1), (-1, -1), 8), # Apply default padding to body rows
        ]))
        story.append(metrics_table)
        
        story.append(Spacer(1, 1*cm))
        
        # ===== 3. FFT 頻譜圖 =====
        story.append(Paragraph("FFT 頻譜分析", styles['subtitle']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
        story.append(Spacer(1, 0.5*cm))
        
        # 計算頻譜
        frequencies, magnitudes = compute_average_spectrum(audio_data, sample_rate)
        magnitudes_weighted = apply_a_weighting(frequencies, magnitudes)
        magnitudes_calibrated = magnitudes_weighted + calibration_offset
        
        # 限制頻率範圍
        mask = (frequencies >= 20) & (frequencies <= 20000)
        freq_plot = frequencies[mask]
        mag_plot = magnitudes_calibrated[mask]
        
        # 創建頻譜圖
        fig_spectrum = go.Figure()
        fig_spectrum.add_trace(go.Scatter(
            x=freq_plot,
            y=mag_plot,
            mode='lines',
            line=dict(color='#1f77b4', width=1),
            name='FFT Spectrum'
        ))
        fig_spectrum.update_layout(
            title='FFT 平均頻譜圖 (A-weighted)',
            xaxis=dict(title='頻率 (Hz)', type='log', range=[np.log10(20), np.log10(20000)]),
            yaxis=dict(title='幅度 dB(A)'),
            template='plotly_white',
            margin=dict(l=60, r=40, t=60, b=60)
        )
        
        # 導出圖片
        spectrum_img = export_plotly_to_image(fig_spectrum, width=800, height=350)
        if spectrum_img:
            story.append(Image(io.BytesIO(spectrum_img), width=16*cm, height=7*cm))
        else:
            story.append(Paragraph("⚠️ 圖表導出失敗", styles['body']))
        
        story.append(PageBreak())
        
        # ===== 4. Level vs Time =====
        story.append(Paragraph("時間-音壓級分析", styles['subtitle']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
        story.append(Spacer(1, 0.5*cm))
        
        # 取得時間剖面
        profile = noise_metrics.get("profile", {})
        if profile:
            times = profile.get("times", [])
            levels = [l + calibration_offset for l in profile.get("levels", [])]
            
            fig_level = go.Figure()
            fig_level.add_trace(go.Scatter(
                x=times,
                y=levels,
                mode='lines',
                line=dict(color='#2ecc71', width=1),
                name='Level'
            ))
            fig_level.update_layout(
                title='Level vs Time',
                xaxis=dict(title='時間 (秒)'),
                yaxis=dict(title='L(A) dB(SPL)'),
                template='plotly_white',
                margin=dict(l=60, r=40, t=60, b=60)
            )
            
            level_img = export_plotly_to_image(fig_level, width=800, height=300)
            if level_img:
                story.append(Image(io.BytesIO(level_img), width=16*cm, height=6*cm))
            else:
                story.append(Paragraph("⚠️ 圖表導出失敗", styles['body']))
        
        story.append(Spacer(1, 1*cm))
        
        # ===== 5. 1/3 倍頻程 =====
        story.append(Paragraph("1/3 倍頻程分析", styles['subtitle']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
        story.append(Spacer(1, 0.5*cm))
        
        # 使用與前端相同的圖表生成函數（確保計算方法一致）
        fig_octave = create_octave_band_chart(audio_data, sample_rate, use_a_weighting=True, calibration_offset=calibration_offset)
        # 調整圖表外觀以適應 PDF
        fig_octave.update_layout(
            title='1/3 倍頻程頻譜 (A-weighted)',
            template='plotly_white',
            margin=dict(l=60, r=40, t=60, b=60)
        )
        
        octave_img = export_plotly_to_image(fig_octave, width=800, height=300)
        if octave_img:
            story.append(Image(io.BytesIO(octave_img), width=16*cm, height=6*cm))
        else:
            story.append(Paragraph("⚠️ 圖表導出失敗", styles['body']))
        
        story.append(PageBreak())
        
        # ===== 6. Discrete Tone 分析 =====
        if analyze_discrete_tone_flag:
            story.append(Paragraph("Discrete Tone 檢測 (ECMA-418-1)", styles['subtitle']))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
            story.append(Spacer(1, 0.5*cm))
            
            tone_result = detect_discrete_tones(audio_data, sample_rate)
            
            if tone_result['tone_detected']:
                story.append(Paragraph(
                    f"⚠️ 偵測到 {len(tone_result['tones'])} 個超過門檻的 Discrete Tone",
                    styles['body']
                ))
                
                # Tone 表格
                tone_data = [["頻率 (Hz)", "PR (dB)", "TNR (dB)", "判定方法", "頻帶"]]
                for tone in tone_result['tones']:
                    tone_data.append([
                        f"{tone['frequency']:.0f}",
                        f"{tone['prominence']:.1f}",
                        f"{tone.get('tnr', 0):.1f}",
                        tone.get('method', 'PR'),
                        tone['band']
                    ])
                
                tone_table = Table(tone_data, colWidths=[3*cm, 2.5*cm, 2.5*cm, 3*cm, 4*cm])
                tone_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e74c3c')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#fdf2f2')),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#333333')),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), font_name),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
                ]))
                story.append(tone_table)
            else:
                story.append(Paragraph(
                    "✓ 未偵測到超過門檻的 Discrete Tone",
                    styles['body']
                ))
            
            story.append(Spacer(1, 0.5*cm))
            story.append(Paragraph(
                f"判定標準: {tone_result.get('criteria', 'ECMA-418-1')}",
                styles['body']
            ))
        
        story.append(Spacer(1, 1*cm))
        
        # ===== 7. SOP 分析結果（支援多模式）=====
        if sop_results:
            story.append(Paragraph("ASUS SOW 高頻音分析", styles['subtitle']))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
            story.append(Spacer(1, 0.5*cm))
            
            # 逐一顯示各模式結果
            for mode_name, result in sop_results.items():
                # 模式標題
                story.append(Paragraph(f"📊 {mode_name} Mode", styles['body']))
                story.append(Spacer(1, 0.3*cm))
                
                # 結果表格
                sop_data = [["項目", "數值"]]
                
                if mode_name == 'IDLE':
                    max_leq = result['max_leq'] + calibration_offset
                    sop_data.append(["Max Leq", f"{max_leq:.1f} dB(A)"])
                    sop_data.append(["SPEC 限制", f"{sop_params.get('idle_spec', 20.0)} dB(A)"])
                    sop_data.append(["判定結果", "PASS ✓" if result['is_pass'] else "FAIL ✗"])
                    
                elif mode_name == 'UE':
                    leq = result['leq'] + calibration_offset
                    sop_data.append(["平均 Leq", f"{leq:.1f} dB(A)"])
                    sop_data.append(["錄音時長", f"{result['duration']:.1f} 秒"])
                    
                elif mode_name == 'Workload':
                    max_leq = result['max_leq'] + calibration_offset
                    sop_data.append(["Max Leq", f"{max_leq:.1f} dB(A)"])
                    sop_data.append(["Fail Rate", f"{result['fail_rate']:.1f}%"])
                    sop_data.append(["判定結果", "PASS ✓" if result['is_pass'] else "FAIL ✗"])
                
                sop_table = Table(sop_data, colWidths=[5*cm, 10*cm])
                sop_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8f9fa')),
                    ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#333333')),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, -1), font_name),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc')),
                ]))
                story.append(sop_table)
                story.append(Spacer(1, 0.3*cm))
                
                # SOP 趨勢圖（IDLE 和 Workload 模式有圖）
                if 'times' in result and 'leqs' in result:
                    times = result['times']
                    leqs = [l + calibration_offset for l in result['leqs']]
                    
                    fig_sop = go.Figure()
                    fig_sop.add_trace(go.Scatter(
                        x=times,
                        y=leqs,
                        mode='lines',
                        line=dict(color='#3498db', width=1),
                        name='Moving Leq'
                    ))
                    
                    # 添加 SPEC 線
                    if mode_name == 'IDLE':
                        spec = sop_params.get('idle_spec', 20.0)
                        fig_sop.add_hline(y=spec, line_dash="dash", line_color="red",
                                          annotation_text=f"SPEC: {spec} dBA")
                    elif mode_name == 'Workload':
                        spec_fail = sop_params.get('work_spec_fail', 22.0)
                        spec_max = sop_params.get('work_spec_max', 28.0)
                        fig_sop.add_hline(y=spec_fail, line_dash="dot", line_color="orange",
                                          annotation_text=f"Fail: {spec_fail} dBA")
                        fig_sop.add_hline(y=spec_max, line_dash="dash", line_color="red",
                                          annotation_text=f"Max: {spec_max} dBA")
                    
                    fig_sop.update_layout(
                        title=f'{mode_name} Mode 趨勢圖',
                        xaxis=dict(title='時間 (秒)'),
                        yaxis=dict(title='音壓級 dB(A)'),
                        template='plotly_white',
                        margin=dict(l=60, r=40, t=60, b=60)
                    )
                    
                    sop_img = export_plotly_to_image(fig_sop, width=800, height=250)
                    if sop_img:
                        story.append(Image(io.BytesIO(sop_img), width=16*cm, height=5*cm))
                    else:
                        story.append(Paragraph("⚠️ 圖表導出失敗", styles['body']))
                
                story.append(Spacer(1, 0.5*cm))
        
        # ===== 頁尾 =====
        story.append(Spacer(1, 2*cm))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc')))
        story.append(Paragraph(
            f"報告生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 聲學測試 AI 分析系統",
            ParagraphStyle('Footer', fontName=font_name, fontSize=9, textColor=colors.gray, alignment=1)
        ))
        
        # 生成 PDF
        doc.build(story)
        
        return buffer.getvalue(), None
        
    except Exception as e:
        import traceback
        return None, f"PDF 生成失敗: {str(e)}\n{traceback.format_exc()}"
