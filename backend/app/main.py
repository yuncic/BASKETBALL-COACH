from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


@app.get("/")
async def root():
    return {"message": "Shooting Analyzer API"}


@app.get("/status")
async def health_check():
    return {"status": "I'm Working!"}