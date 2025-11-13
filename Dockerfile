# Python 3.10 사용 (PyTorch 호환성)
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치 (OpenCV headless에 필요)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 백엔드 코드 복사
COPY backend/ ./backend/

# 프론트엔드 복사
COPY frontend/ ./frontend/

# 작업 디렉토리를 backend로 변경
WORKDIR /app/backend

# 포트 노출
EXPOSE 10000

# 환경 변수 설정
ENV PYTHONPATH=/app/backend
ENV OPENCV_DISABLE_OPENCL=1
ENV QT_QPA_PLATFORM=offscreen

# Railway는 PORT 환경 변수를 자동으로 설정하므로 이를 사용
# 기본값은 10000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}"]

