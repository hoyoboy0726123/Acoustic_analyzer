# -*- coding: utf-8 -*-
"""
聲學測試 AI 分析系統 - 報告生成模組

功能 (AUD-008):
- 整合所有分析結果
- 產生不同格式報告 (JSON, Markdown, HTML)
- 提供測試結果摘要
"""

from typing import Dict, Any, Optional
import json
from datetime import datetime
from pathlib import Path

# 從 config 導入設定
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_full_report(
    file_info: Dict[str, Any],
    noise_level: Dict[str, Any],
    fft_analysis: Dict[str, Any],
    discrete_tone: Dict[str, Any],
    high_freq_analysis: Dict[str, Any],
    band_analysis: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """產生完整測試報告

    整合所有分析結果，產生結構化的測試報告。

    Args:
        file_info: 音檔資訊
        noise_level: 噪音等級分析結果
        fft_analysis: FFT 分析結果
        discrete_tone: Discrete Tone 檢測結果
        high_freq_analysis: 高頻分析結果
        band_analysis: 頻帶分析結果 (可選)

    Returns:
        Dict: 完整測試報告
    """
    # 產生報告時間戳記
    timestamp = datetime.now().isoformat()

    # 判定整體結果
    overall_result = _determine_overall_result(
        noise_level, discrete_tone, high_freq_analysis
    )

    report = {
        "report_info": {
            "title": "聲學測試分析報告",
            "generated_at": timestamp,
            "version": "1.0.0"
        },
        "file_info": file_info,
        "results": {
            "noise_level": noise_level,
            "fft_analysis": _summarize_fft(fft_analysis),
            "discrete_tone": discrete_tone,
            "high_frequency": high_freq_analysis
        },
        "overall": {
            "result": overall_result["status"],
            "score": overall_result["score"],
            "summary": overall_result["summary"],
            "issues": overall_result["issues"],
            "recommendations": overall_result["recommendations"]
        }
    }

    if band_analysis:
        report["results"]["band_analysis"] = band_analysis

    return report


def _determine_overall_result(
    noise_level: Dict[str, Any],
    discrete_tone: Dict[str, Any],
    high_freq: Dict[str, Any]
) -> Dict[str, Any]:
    """判定整體測試結果"""
    issues = []
    recommendations = []
    score = 100

    # 檢查噪音等級
    leq = noise_level.get("leq_dba", 0)
    if leq > -20:  # 相對 dB，實際應依規格調整
        score -= 10
        issues.append(f"噪音等級 Leq = {leq:.1f} dB")

    # 檢查 Discrete Tone
    if discrete_tone.get("tone_detected", False):
        score -= 20
        tones = discrete_tone.get("tones", [])
        for tone in tones[:3]:  # 最多列出 3 個
            issues.append(
                f"Discrete Tone 於 {tone['frequency']:.0f} Hz "
                f"(突出量 {tone['prominence']:.1f} dB)"
            )
        recommendations.append("建議調查 Discrete Tone 來源")

    # 檢查電感嘯叫
    if high_freq.get("coil_whine_detected", False):
        score -= 25
        freq = high_freq.get("coil_whine_frequency", 0)
        issues.append(f"偵測到電感嘯叫 ({freq:.0f} Hz)")
        recommendations.append("建議檢查電源管理元件")

    # 檢查高頻狀態
    hf_status = high_freq.get("overall_status", "PASS")
    if hf_status == "FAIL":
        score -= 15
        issues.append("高頻分析結果異常")
    elif hf_status == "WARNING":
        score -= 5
        issues.append("高頻分析結果需注意")

    # 確保分數在 0-100 範圍
    score = max(0, min(100, score))

    # 判定狀態
    if score >= 80:
        status = "PASS"
        summary = "測試通過，無明顯異常"
    elif score >= 60:
        status = "WARNING"
        summary = "測試合格但有潛在問題"
    else:
        status = "FAIL"
        summary = "測試未通過，需進一步調查"

    if not issues:
        issues = ["無異常發現"]
    if not recommendations:
        recommendations = ["維持目前設計"]

    return {
        "status": status,
        "score": score,
        "summary": summary,
        "issues": issues,
        "recommendations": recommendations
    }


def _summarize_fft(fft_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """摘要 FFT 分析結果"""
    return {
        "max_frequency": fft_analysis.get("max_frequency"),
        "max_magnitude": fft_analysis.get("max_magnitude"),
        "peak_count": len(fft_analysis.get("peak_frequencies", [])),
        "band_energies": fft_analysis.get("band_energies", {})
    }


def generate_summary_report(full_report: Dict[str, Any]) -> str:
    """產生簡易摘要報告

    產生簡潔的文字摘要。

    Args:
        full_report: generate_full_report 的輸出

    Returns:
        str: 摘要文字
    """
    overall = full_report.get("overall", {})
    file_info = full_report.get("file_info", {})

    lines = [
        "=" * 50,
        "聲學測試分析報告摘要",
        "=" * 50,
        "",
        f"測試時間: {full_report.get('report_info', {}).get('generated_at', 'N/A')}",
        f"檔案名稱: {file_info.get('file_name', 'N/A')}",
        "",
        f"整體結果: {overall.get('result', 'N/A')} (分數: {overall.get('score', 0)})",
        f"摘要: {overall.get('summary', 'N/A')}",
        "",
        "發現問題:",
    ]

    for issue in overall.get("issues", []):
        lines.append(f"  - {issue}")

    lines.append("")
    lines.append("建議:")

    for rec in overall.get("recommendations", []):
        lines.append(f"  - {rec}")

    lines.append("")
    lines.append("=" * 50)

    return "\n".join(lines)


def generate_markdown_report(full_report: Dict[str, Any]) -> str:
    """產生 Markdown 格式報告

    Args:
        full_report: generate_full_report 的輸出

    Returns:
        str: Markdown 格式報告
    """
    report_info = full_report.get("report_info", {})
    file_info = full_report.get("file_info", {})
    overall = full_report.get("overall", {})
    results = full_report.get("results", {})

    md = f"""# 聲學測試分析報告

**產生時間:** {report_info.get('generated_at', 'N/A')}

---

## 📁 檔案資訊

| 項目 | 數值 |
|------|------|
| 檔案名稱 | {file_info.get('file_name', 'N/A')} |
| 取樣率 | {file_info.get('sample_rate', 'N/A')} Hz |
| 長度 | {file_info.get('duration', 0):.2f} 秒 |
| 格式 | {file_info.get('format', 'N/A')} |

---

## 📊 整體結果

**狀態:** {overall.get('result', 'N/A')}  
**分數:** {overall.get('score', 0)}/100  
**摘要:** {overall.get('summary', 'N/A')}

### ⚠️ 發現問題

"""

    for issue in overall.get("issues", []):
        md += f"- {issue}\n"

    md += """
### 💡 建議

"""

    for rec in overall.get("recommendations", []):
        md += f"- {rec}\n"

    # 噪音等級
    noise = results.get("noise_level", {})
    md += f"""
---

## 🔊 噪音等級分析

| 指標 | 數值 |
|------|------|
| Leq | {noise.get('leq_dba', 'N/A')} dB(A) |
| Lmax | {noise.get('lmax_dba', 'N/A')} dB(A) |
| Lmin | {noise.get('lmin_dba', 'N/A')} dB(A) |
| L10 | {noise.get('l10', 'N/A')} dB(A) |
| L90 | {noise.get('l90', 'N/A')} dB(A) |

"""

    # Discrete Tone
    dt = results.get("discrete_tone", {})
    md += f"""
---

## 🎵 Discrete Tone 檢測

**偵測結果:** {'⚠️ 偵測到 Tone' if dt.get('tone_detected') else '✅ 未偵測到'}

"""

    if dt.get("tones"):
        md += "| 頻率 (Hz) | 突出量 (dB) | 門檻 (dB) | 頻帶 |\n"
        md += "|-----------|-------------|-----------|------|\n"
        for tone in dt.get("tones", []):
            md += f"| {tone.get('frequency', 0):.0f} | {tone.get('prominence', 0):.1f} | {tone.get('threshold', 0):.0f} | {tone.get('band', 'N/A')} |\n"

    # 高頻分析
    hf = results.get("high_frequency", {})
    md += f"""
---

## ⚡ 高頻音隔離分析

**整體狀態:** {hf.get('overall_status', 'N/A')}  
**電感嘯叫:** {'⚠️ 偵測到' if hf.get('coil_whine_detected') else '✅ 未偵測到'}

"""

    if hf.get("coil_whine_detected"):
        md += f"""
**電感嘯叫詳情:**
- 頻率: {hf.get('coil_whine_frequency', 0):.0f} Hz
- 突出量: {hf.get('coil_whine_prominence', 0):.1f} dB
- 可能原因: {hf.get('possible_cause', 'N/A')}

"""

    md += f"""
**建議:** {hf.get('recommendation', 'N/A')}

---

*報告由聲學測試 AI 分析系統 v{report_info.get('version', '1.0.0')} 自動產生*
"""

    return md


def save_report(
    report: Dict[str, Any],
    output_path: str,
    format: str = "json"
) -> str:
    """儲存報告至檔案

    Args:
        report: 報告內容
        output_path: 輸出路徑 (不含副檔名)
        format: 格式 ("json", "md", "txt")

    Returns:
        str: 儲存的檔案路徑
    """
    output_path = Path(output_path)

    if format == "json":
        file_path = output_path.with_suffix(".json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    elif format == "md":
        file_path = output_path.with_suffix(".md")
        md_content = generate_markdown_report(report)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

    elif format == "txt":
        file_path = output_path.with_suffix(".txt")
        txt_content = generate_summary_report(report)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(txt_content)

    else:
        raise ValueError(f"不支援的格式: {format}")

    return str(file_path)
