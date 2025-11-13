from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.routes import analyze

app = FastAPI(title="Shooting Analyzer API", version="1.0.0")

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
    return {"status": "healthy"}


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
print(f"   프론트엔드 준비됨: {FRONTEND_DIR is not None and INDEX_FILE.exists() if FRONTEND_DIR else False}")


def _frontend_ready() -> bool:
    return FRONTEND_DIR is not None and INDEX_FILE.exists()


if _frontend_ready():

    @app.get("/", include_in_schema=False)
    async def serve_frontend_root():
        return FileResponse(INDEX_FILE)

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_frontend_assets(full_path: str):
        # API 경로는 제외
        if full_path.startswith("api/"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not found")
        
        target_path = FRONTEND_DIR / full_path
        if target_path.is_file() and target_path.exists():
            return FileResponse(target_path)
        # 파일이 없으면 index.html 반환 (SPA 라우팅 지원)
        return FileResponse(INDEX_FILE)

else:

    @app.get("/", include_in_schema=False)
    async def frontend_missing():
        return JSONResponse(
            content={
                "message": "Shooting Analyzer API",
                "status": "running",
                "warning": "frontend 디렉토리를 찾을 수 없습니다.",
                "api_docs": "/docs"
            },
            status_code=200
        )
