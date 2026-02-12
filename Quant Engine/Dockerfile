FROM python:3.9-slim

WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装基础依赖
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 复制源代码 (V10.0: Copy entire api package)
COPY api/ api/

# 暴露端口
EXPOSE 8080

# 启动命令 (V10.0: Run as module)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
