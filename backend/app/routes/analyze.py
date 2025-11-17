import os
import json
import tempfile
import base64
import urllib.parse
import traceback
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, Request
from fastapi.responses import Response, JSONResponse

from app.services.analyze_service import analyze_video_from_path

router = APIRouter(prefix="/api", tags=["analyze"])


@router.options("/report")
async def options_report():
    return Response(status_code=200)


@router.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...)):
    """비디오 파일을 업로드하고 분석하여 결과 비디오와 리포트를 반환"""
    in_path = None
    out_path = None
    report_path = None
    
    try:
        # User-Agent로 모바일 감지
        user_agent = request.headers.get("user-agent", "").lower() if request else ""
        is_mobile = any(keyword in user_agent for keyword in ["mobile", "android", "iphone", "ipad", "ipod"])
        
        suffix = Path(file.filename).suffix if file.filename else ".mp4"
        in_path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
        print(f"📁 입력 파일 저장: {in_path}")
        print(f"📱 User-Agent: {user_agent[:100] if user_agent else 'None'}")
        print(f"📱 모바일 감지: {is_mobile}")
        
        with open(in_path, "wb") as f:
            content = await file.read()
            f.write(content)
        print(f"✅ 파일 업로드 완료: {len(content)} bytes")

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        print(f"🔍 영상 분석 시작: {in_path} -> {out_path}")
        
        report = analyze_video_from_path(in_path, out_path, is_mobile=is_mobile)
        print("🧩 report 결과:", json.dumps(report, ensure_ascii=False, indent=2))

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError("분석된 영상 파일이 생성되지 않았습니다.")

        report_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
        with open(report_path, "w", encoding="utf-8") as rf:
            json.dump(report, rf, ensure_ascii=False, indent=2)

        # 🔹 report를 Base64로 인코딩해서 헤더에도 같이 포함
        encoded = base64.b64encode(json.dumps(report, ensure_ascii=False).encode("utf-8")).decode("utf-8")

        with open(out_path, "rb") as f:
            video_data = f.read()

        response = Response(content=video_data, media_type="video/mp4")
        response.headers["X-Report-Path"] = report_path
        response.headers["X-Report-Base64"] = encoded
        response.headers["Access-Control-Expose-Headers"] = "x-report-path, X-Report-Path, x-report-base64, X-Report-Base64"
        response.headers["Access-Control-Allow-Headers"] = "x-report-path, X-Report-Path, x-report-base64, X-Report-Base64"

        return response
        
    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        print(f"❌ 분석 중 오류 발생:")
        print(f"   에러 메시지: {error_msg}")
        print(f"   전체 스택 트레이스:\n{error_trace}")
        
        # 임시 파일 정리
        for path in [in_path, out_path, report_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass
        
        # 에러 응답 반환
        error_response = {
            "error": error_msg,
            "detail": error_trace if os.getenv("DEBUG", "false").lower() == "true" else None
        }
        return JSONResponse(
            content=error_response,
            status_code=500
        )


@router.get("/report")
async def get_report(path: str = Query(..., description="리포트 파일의 절대경로")):
    """리포트 파일 경로를 받아 리포트 데이터를 반환"""
    path = urllib.parse.unquote(path)
    print("📁 요청받은 리포트 경로:", path)

    if not os.path.exists(path):
        print("❌ 리포트 파일 없음:", path)
        error_data = {"error": f"리포트 없음: {path}"}
        json_str = json.dumps(error_data, ensure_ascii=False)
        return Response(
            content=json_str.encode("utf-8"),
            media_type="application/json",
            headers={"Content-Type": "application/json; charset=utf-8"},
            status_code=404
        )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("✅ 리포트 반환 완료")

    # JSONResponse 대신 Response를 사용하여 ensure_ascii=False 명시
    json_str = json.dumps(data, ensure_ascii=False)
    return Response(
        content=json_str.encode("utf-8"),
        media_type="application/json",
        headers={"Content-Type": "application/json; charset=utf-8"}
    )

