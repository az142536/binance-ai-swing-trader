FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 設定環境變數防止 Python 緩衝輸出
ENV PYTHONUNBUFFERED=1

# 安裝相依套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案程式碼
COPY . .

# 預設啟動命令 (可自行切換為 paper 或 live)
CMD ["python", "main.py", "paper", "--symbol", "SUIUSDT", "--interval", "1h"]

