# Python 3.10 사용 (PyTorch 호환성)
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치 (OpenCV headless에 필요)
# 폰트 설치 (한글 폰트 포함)
# ffmpeg 설치 (비디오 재인코딩용)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    fonts-dejavu-core \
    fonts-noto-cjk \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "torch==2.5.1" "torchvision==0.20.1" --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    rm -rf /root/.cache/pip /tmp/* /var/tmp/*

# 백엔드 코드 복사
COPY backend/ ./backend/

# 프론트엔드 복사
COPY frontend/ ./frontend/

# 작업 디렉토리를 backend로 변경
WORKDIR /app/backend

# start 스크립트 실행 권한 부여
RUN chmod +x start.sh start.py

# 포트 노출
EXPOSE 10000

# 환경 변수 설정
ENV PYTHONPATH=/app/backend
ENV OPENCV_DISABLE_OPENCL=1
ENV QT_QPA_PLATFORM=offscreen

# Railway는 PORT 환경 변수를 자동으로 설정하므로 이를 사용
# Python 스크립트 사용 (더 확실함)
CMD ["python", "start.py"]

