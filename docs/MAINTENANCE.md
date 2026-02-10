# AkShare Quant Engine - 运维手册

> **⚠️ 内部文档 - 禁止上传到公开仓库**  
> 最后更新: 2026-02-07 22:24

---

## 目录

1. [项目概览](#1-项目概览)
2. [架构说明](#2-架构说明)
3. [环境配置](#3-环境配置)
4. [部署指南](#4-部署指南)
5. [API 文档](#5-api-文档)
6. [工作流配置](#6-工作流配置)
7. [飞书集成](#7-飞书集成)
8. [故障排查](#8-故障排查)
9. [维护清单](#9-维护清单)

---

## 1. 项目概览

### 1.1 项目信息

| 项目 | 值 |
|------|-----|
| 项目名称 | AkShare Quant Engine |
| 版本 | V9.1 Titan + V8.0 Evolution |
| GitHub 仓库 | `MINGCHOW/akshare-quant-engine` |
| 本地路径 | `d:\Antigravity-` |
| 云端 API | `https://jpthermjdexc.ap-northeast-1.clawcloudrun.com` |

### 1.2 核心功能

| 功能 | 描述 |
|------|------|
| 股票数据获取 | A股/港股实时数据、历史K线 |
| 技术分析 | ATR、RSI、BIAS、均线形态 |
| 信号生成 | 买入/卖出/观望信号 |
| 投后管理 | 持仓监控、止损/止盈跟踪 |
| 通知推送 | 飞书多维表格 + 消息卡片 |

### 1.3 技术栈

| 组件 | 技术 |
|------|------|
| 后端 | Python 3.11 + FastAPI |
| 数据源 | AkShare (主) / PyTDX / BaoStock (备用) |
| 工作流 | n8n (自托管) |
| 通知 | 飞书多维表格 + 飞书机器人 |
| 云端 | ClawCloud Run (Docker) |

---

## 2. 架构说明

### 2.1 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                         n8n 工作流                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ 定时触发器   │  │ HTTP Request │  │ 飞书节点         │  │
│  │ (Schedule)   │→ │ (API Call)   │→ │ (Write/Notify)   │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI 后端                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ /market      │  │ /analyze_full│  │ /check_positions │  │
│  │ 大盘状态     │  │ 股票分析     │  │ 持仓检查         │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     数据层                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ AkShare      │  │ PyTDX        │  │ BaoStock         │  │
│  │ (Primary)    │  │ (Fallback)   │  │ (Historical)     │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 文件结构

# Antigravity Quant Engine (V10.0 Titan) - 维护手册

## 📂 项目结构 (V10.0)

- **API 服务**: `api/main.py` (FastAPI 入口), `api/fetcher.py` (数据层), `api/quant.py` (量化层)
- **核心工作流**: `workflow/stock_analysis.json` (AI 分析)
- **监控工作流**: `workflow/monitor_position.json` (持仓), `workflow/monitor_heartbeat.json` (心跳)
- **部署配置**: `Dockerfile`, `requirements.txt`
- **测试**: `tests/test_quant.py`

## 🚀 部署与更新

### 常规更新流程
1. **修改代码**: 在 `api/` 目录下修改逻辑
2. **本地测试**: 运行 `pytest tests/` 确保无逻辑错误
3. **提交代码**: `git commit -am "fix: description" && git push`
4. **自动部署**: ClawCloud 会自动拉取并重建容器 (需配置 CI/CD 或手动触发)

### API Key 管理
- 生产环境 Key: `aqe-k8x7m2pQ9vR4wL6nJ3sY5tB1` (V10.0 Default)
- 在 ClawCloud 环境变量中设置 `API_KEY` 可覆盖默认值。
- 调用示例:
  ```bash
  curl -H "X-API-Key: $API_KEY" https://your-api-url/health
  ```

---

## 3. 环境配置

### 3.1 本地开发环境

```bash
# Python 版本
Python 3.11+

# 创建虚拟环境
python -m venv venv
.\venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 启动本地服务
cd api
uvicorn akshare_api:app --host 0.0.0.0 --port 8000 --reload
```

### 3.2 环境变量

| 变量名 | 用途 | 示例值 |
|--------|------|--------|
| `PORT` | API 端口 | `8000` |
| `LOG_LEVEL` | 日志级别 | `INFO` |

### 3.3 依赖版本

```
fastapi>=0.100.0
uvicorn>=0.22.0
akshare>=1.10.0
pandas>=2.0.0
numpy>=1.24.0
pydantic>=2.0.0
```

---

## 4. 部署指南

### 4.1 ClawCloud Run 部署

#### 4.1.1 首次部署

1. 登录 ClawCloud Console: https://console.clawcloud.cn/
2. 创建新应用 → 选择 "Git Repository"
3. 连接 GitHub 仓库: `MINGCHOW/akshare-quant-engine`
4. 配置构建:
   - Runtime: Docker
   - Dockerfile 路径: `./Dockerfile`
   - 端口: 8000
5. 选择区域: `ap-northeast-1` (日本东京)
6. 点击 Deploy

#### 4.1.2 更新部署

**自动部署** (如已配置):
- Push 到 `main` 分支后自动触发

**手动部署**:
1. 进入 ClawCloud Console
2. 选择应用 → Redeploy

#### 4.1.3 部署验证

```bash
# 健康检查
curl https://jpthermjdexc.ap-northeast-1.clawcloudrun.com/health

# 预期响应
{
  "status": "healthy",
  "timestamp": "...",
  "latency_ms": xxx,
  "checks": {...}
}
```

### 4.2 Docker 本地构建

```bash
# 构建镜像
docker build -t akshare-quant .

# 运行容器
docker run -p 8000:8000 akshare-quant
```

### 4.3 n8n 工作流部署

1. 打开 n8n 界面
2. 导入工作流: Settings → Import from File
3. 选择 `workflow/` 目录下的 JSON 文件
4. 配置凭据 (见第7节)
5. 激活工作流

---

## 5. API 文档

### 5.1 端点列表

#### GET /health
```json
// 请求
GET /health

// 响应
{
  "status": "healthy",
  "timestamp": "2026-02-07T22:00:00",
  "latency_ms": 4500,
  "checks": {
    "data_source": {"status": "ok", "rows": 8334},
    "circuit_breaker": {"error_count": 0, "is_open": false}
  }
}
```

#### POST /health/reset
```json
// 请求
POST /health/reset

// 响应
{"message": "Breaker reset", "error_count": 0}
```

#### GET /market
```json
// 请求
GET /market

// 响应
{
  "market_status": "Normal|Caution|Bear|Crash",
  "index_price": 3200.50,
  "ma20": 3150.00,
  "up_count": 2500,
  "down_count": 1800,
  "flat_count": 200,
  "is_frozen": false
}
```

#### POST /analyze_full
```json
// 请求
POST /analyze_full
{
  "code": "000001",
  "balance": 100000,
  "risk": 0.02
}

// 响应
{
  "code": "000001",
  "signal_type": "买入|观望|卖出",
  "current_price": 15.50,
  "stop_loss": 14.50,
  "take_profit": 17.00,
  "suggested_position": 500,
  ...
}
```

#### POST /check_positions
```json
// 请求
POST /check_positions
{
  "positions": [
    {
      "code": "601958",
      "market": "CN",
      "buy_price": 19.78,
      "current_stop": 18.00,
      "target_price": 22.00,
      "shares": 10,
      "record_id": "recXXXXXX"
    }
  ]
}

// 响应
{
  "positions": [
    {
      "code": "601958",
      "current_price": 19.56,
      "action": "HOLD|SELL_STOP|SELL_TARGET",
      "reason": "继续持有",
      "pnl_percent": -1.11,
      "pnl_amount": -220.00,
      "new_stop": 18.19,
      "record_id": "recXXXXXX"
    }
  ]
}
```

### 5.2 错误处理

| HTTP 状态码 | 含义 |
|-------------|------|
| 200 | 成功 |
| 422 | 参数验证失败 |
| 500 | 服务器内部错误 |
| 503 | 熔断器打开 (数据源不可用) |

---

## 6. 工作流配置

### 6.1 AH Stock V9.1 Titan

| 配置项 | 值 |
|--------|-----|
| 触发器 | 每日 09:20 / 09:35 / 09:50 (工作日) |
| 股票列表来源 | 飞书多维表格【每日分析】 |
| 输出目标 | 飞书多维表格【分析记录】 |
| 并发处理 | 每股间隔 3 秒 |

### 6.2 V8_Heartbeat_Monitor

| 配置项 | 值 |
|--------|-----|
| 触发器 | 每小时整点 |
| 检查目标 | `/health` 端点 |
| 通知条件 | status ≠ "healthy" |
| 通知方式 | 红色卡片 (异常) / 绿色卡片 (正常) |

### 6.3 V8_Position_Monitor

| 配置项 | 值 |
|--------|-----|
| 触发器 | 每日 15:10 (工作日) |
| 数据来源 | 飞书【持仓管理】表 |
| 筛选条件 | 持仓状态 = "持仓" |
| 回写字段 | 当前止损, 盈亏比例, 盈亏金额, 现价 |

---

## 7. 飞书集成

### 7.1 凭据配置

⚠️ **以下为敏感信息，仅限内部使用**

| 配置项 | 值 |
|--------|-----|
| n8n 节点类型 | `n8n-nodes-feishu-lite.feishuNode` |
| 凭据 ID | `RcP0KB4O5l2Y95Bs` |
| 凭据名称 | `Feishu account 2` |

### 7.2 多维表格配置

#### 【分析记录】表

| 配置项 | 值 |
|--------|-----|
| App Token | `RVghbRvYgacqs3s82qkcl83bn7e` |
| Table ID | `tblvrNDNrjAZwBZc` |
| URL | https://xcnf59usubzt.feishu.cn/base/RVghbRvYgacqs3s82qkcl83bn7e |

#### 【持仓管理】表

| 配置项 | 值 |
|--------|-----|
| App Token | `RVghbRvYgacqs3s82qkcl83bn7e` |
| Table ID | `tblKGT8D4rDur8Gi` |
| URL | https://xcnf59usubzt.feishu.cn/base/RVghbRvYgacqs3s82qkcl83bn7e?table=tblKGT8D4rDur8Gi |

**关键字段说明**:
- `买入手数`: **实际为“持仓股数”** (例如 200 代表 200 股，不是 200 手)
- `当前止损`: 动态更新的止损价
- `盈亏金额`: `(现价 - 买入价) * 持仓股数`

### 7.3 消息通知配置

| 配置项 | 值 |
|--------|-----|
| receive_id_type | `user_id` |
| receive_id | `bg99a8dc` |

---

## 8. 故障排查

### 8.1 常见问题

#### API 返回 500 错误

**可能原因**:
- 数据源暂时不可用
- 股票代码格式错误

**排查步骤**:
1. 检查 `/health` 端点
2. 查看容器日志
3. 确认熔断器状态

#### 飞书节点报错

**可能原因**:
- 凭据过期
- 字段名不匹配

**排查步骤**:
1. 检查凭据是否有效
2. 确认飞书表字段名完全匹配
3. 检查字段类型 (文本/数字)

#### 止损/止盈误触发

**可能原因**:
- 止损价或目标价为 0

**解决方案**:
- API 已添加 `> 0` 判断，确保部署最新版本

### 8.2 日志查看

```bash
# ClawCloud Run 日志
登录 Console → 选择应用 → Logs

# 本地日志
uvicorn 输出到控制台
```

---

## 9. 维护清单

### 9.1 日常检查 (每日)

- [ ] 确认 15:10 投后盯盘工作流正常执行
- [ ] 检查飞书通知是否收到
- [ ] 查看【持仓管理】表字段是否更新

### 9.2 周检查

- [ ] 检查 `/health` 端点响应时间
- [ ] 确认熔断器错误计数归零
- [ ] 检查云端容器资源使用

### 9.3 月度维护

- [ ] 更新 Python 依赖 (`pip install --upgrade -r requirements.txt`)
- [ ] 检查 AkShare 版本更新
- [ ] 备份飞书多维表格数据
- [ ] 清理过期的分析记录

### 9.4 版本更新检查清单

- [ ] 本地测试通过
- [ ] Git commit 并 push
- [ ] 确认云端自动部署或手动 Redeploy
- [ ] 验证 `/health` 返回正确版本
- [ ] 测试工作流执行

### 9.5 数据备份

| 数据 | 备份方式 | 频率 |
|------|----------|------|
| 飞书表数据 | 导出 Excel | 每周 |
| 工作流配置 | 导出 JSON | 每次修改后 |
| 代码 | Git | 持续 |

---

## 附录

### A. Git 命令速查

```bash
# 查看状态
git status

# 提交更改
git add . && git commit -m "描述"

# 推送到远程
git push origin main

# 查看历史
git log -n 5 --oneline
```

### B. API 快速测试

```bash
# PowerShell (使用 -UseBasicParsing 避免警告)
Invoke-WebRequest -Uri "https://jpthermjdexc.ap-northeast-1.clawcloudrun.com/health" -UseBasicParsing

# 或使用 curl.exe
curl.exe https://jpthermjdexc.ap-northeast-1.clawcloudrun.com/health
```

### C. 紧急联系

如遇紧急问题，检查以下资源:
- ClawCloud 状态: https://status.clawcloud.cn/
- AkShare 文档: https://akshare.akfamily.xyz/
- 飞书开放平台: https://open.feishu.cn/

---

> **文档维护**: 请在每次重大更新后同步更新本文档
