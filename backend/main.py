from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.routes import analyze

app = FastAPI(title="Shooting Analyzer API", version="1.0.0")

# 서버 시작 시점에 경로 확인 (즉시 실행)
import sys
print("=" * 50)
print("🚀 FastAPI 앱 초기화 중...")
print(f"Python 버전: {sys.version}")
print("=" * 50)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Report-Path", "x-report-path", "X-Report-Base64", "x-report-base64"],
)

# 라우터 등록
app.include_router(analyze.router)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Server is running"}

@app.get("/api/health")
async def api_health_check():
    return {"status": "healthy", "message": "API is running"}


# ----- 정적 프론트엔드 제공 -----
# Docker 컨테이너에서: /app/backend/main.py
# 따라서 /app/frontend를 찾아야 함
_current_file = Path(__file__).resolve()  # /app/backend/main.py
_app_dir = _current_file.parent  # /app/backend
_project_root = _app_dir.parent  # /app

_frontend_candidates = [
    _project_root / "frontend",  # /app/frontend (Docker에서)
    _app_dir / "frontend",       # /app/backend/frontend (대체 경로)
]

FRONTEND_DIR = next((path for path in _frontend_candidates if path.exists()), None)
INDEX_FILE = FRONTEND_DIR / "index.html" if FRONTEND_DIR else None

# 디버깅: 경로 확인
print(f"🔍 Frontend 경로 확인:")
print(f"   현재 파일: {_current_file}")
print(f"   프로젝트 루트: {_project_root}")
print(f"   프론트엔드 후보: {[str(p) for p in _frontend_candidates]}")
print(f"   찾은 프론트엔드: {FRONTEND_DIR}")
print(f"   index.html: {INDEX_FILE}")
if FRONTEND_DIR:
    print(f"   프론트엔드 디렉토리 존재: {FRONTEND_DIR.exists()}")
    if FRONTEND_DIR.exists():
        print(f"   프론트엔드 파일 목록: {list(FRONTEND_DIR.iterdir())[:10]}")
if INDEX_FILE:
    print(f"   index.html 존재: {INDEX_FILE.exists()}")
print(f"   프론트엔드 준비됨: {FRONTEND_DIR is not None and INDEX_FILE.exists() if FRONTEND_DIR else False}")


def _frontend_ready() -> bool:
    return FRONTEND_DIR is not None and INDEX_FILE.exists()


# 루트 경로는 항상 응답하도록 설정
@app.get("/", include_in_schema=False)
async def serve_root():
    if _frontend_ready():
        return FileResponse(INDEX_FILE)
    else:
        return JSONResponse(
            content={
                "message": "Shooting Analyzer API",
                "status": "running",
                "warning": "frontend 디렉토리를 찾을 수 없습니다.",
                "api_docs": "/docs",
                "health": "/health"
            },
            status_code=200
        )

# 프론트엔드 정적 파일 제공
@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend_assets(full_path: str):
    # API 경로는 제외
    if full_path.startswith("api/"):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not found")
    
    # 프론트엔드가 준비된 경우에만 파일 제공
    if _frontend_ready():
        target_path = FRONTEND_DIR / full_path
        if target_path.is_file() and target_path.exists():
            return FileResponse(target_path)
        # 파일이 없으면 index.html 반환 (SPA 라우팅 지원)
        return FileResponse(INDEX_FILE)
    else:
        # 프론트엔드가 없으면 API 정보 반환
        return JSONResponse(
            content={
                "message": "Shooting Analyzer API",
                "status": "running",
                "path": full_path,
                "warning": "frontend 디렉토리를 찾을 수 없습니다.",
                "api_docs": "/docs"
            },
            status_code=404
        )
