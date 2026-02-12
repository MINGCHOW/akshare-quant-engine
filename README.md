# Antigravity Quant Engine (V10.0)

> **Enterprise-Grade AI Quantitative Analysis System**  
> *Formerly AkShare-Quant-Engine*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)

A modular, multi-source quantitative trading engine integrating **AkShare**, **Tencent**, **Yahoo Finance**, and **AI Agents** to provide real-time market analysis, signal generation, and automated risk control.

---

## 🚀 Key Features (V10.0 Titan)

### 1. Multi-Source Data Fetching (8-Layer Shield)
- **Resilient Architecture**: Auto-fallback across 8 data sources (AkShare → Tencent → Yahoo → Baostock → etc.)
- **Cross-Market**: Full support for **A-share** and **HK stocks**.
- **Anti-Scraping**: Intelligent retry logic, dynamic headers, and circuit breakers.

### 2. Advanced Quantitative Core
- **Dual-Market Signals**: Customized strategies for CN (A-share) and HK markets.
- **Titan V10 Algorithm**: 
  - **Symmetric Scoring**: Balanced Buy/Sell signal generation (0-100 scale).
  - **Dynamic Risk Control**: ATR-driven Stop Loss/Take Profit (2:1 Reward/Risk ratio).
  - **MACD Integration**: Trend confirmation with Golden/Death cross detection.
- **ETF Precision**: Exact detection logic for HK/CN ETFs vs individual stocks.

### 3. Workflow Automation (n8n)
- **AI Analysis**: LLM-driven report generation (Gemini/DeepSeek) integrated via n8n.
- **Notifications**: Real-time Feishu/Lark cards with color-coded signals.
- **Monitoring**: Automated Heartbeat and Position Monitoring workflows.

---

## 📂 Project Structure

```text
D:\ANTIGRAVITY-QUANT-ENGINE
├─ api/
│  ├─ main.py       # API Entry Point (FastAPI)
│  ├─ fetcher.py    # Data Layer (8-Layer Retry)
│  ├─ quant.py      # Quant Core (Indicators/Signals)
│  └─ __init__.py
├─ workflow/
│  ├─ stock_analysis.json      # Main AI Analysis Workflow
│  ├─ monitor_heartbeat.json   # System Health Check
│  └─ monitor_position.json    # Position Risk Monitor
├─ tests/
│  └─ test_quant.py # Unit Tests
├─ Dockerfile       # Cloud Run / Docker Deployment
└─ requirements.txt # Pinned Dependencies
```

---

## 🛠️ Deployment

### Docker (Recommended)

```bash
# Build & Run
docker build -t ag-quant-engine .
docker run -p 8080:8080 -e API_KEY=your_secure_key ag-quant-engine
```

### API Usage

- **Health Check**: `GET /health`
- **Market Status**: `GET /market`
- **Full Analysis**: `POST /analyze_full` (Requires `X-API-Key`)

---

## 🛡️ Security & Privacy

- **API Authentication**: All critical endpoints protected via `X-API-Key`.
- **Privacy First**: Sensitive credentials (workflow secrets) are strictly separated from codebase via Templates.

---

## ⚖️ Disclaimer

This project is for **research and educational purposes only**. Quantitative trading involves significant financial risk. Use at your own risk.
