from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

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
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_frontend_candidates = [
    PROJECT_ROOT / "frontend",                         # 최상위 frontend/
    Path(__file__).resolve().parent.parent / "frontend",  # backend/frontend/ (대체 경로)
]

FRONTEND_DIR = next((path for path in _frontend_candidates if path.exists()), None)
INDEX_FILE = FRONTEND_DIR / "index.html" if FRONTEND_DIR else None


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
        return {
            "message": "Shooting Analyzer API",
            "warning": "frontend 디렉토리를 찾을 수 없습니다. README를 참고해 프론트엔드를 준비하세요.",
        }
