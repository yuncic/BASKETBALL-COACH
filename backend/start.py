#!/usr/bin/env python3
"""Railway 배포용 시작 스크립트"""
import os
import sys

# PORT 환경 변수 읽기 (Railway가 자동 설정)
port_str = os.environ.get("PORT", "10000")
try:
    port = int(port_str)
except ValueError:
    print(f"Warning: Invalid PORT value '{port_str}', using default 10000")
    port = 10000

host = "0.0.0.0"

print(f"🚀 Starting server on {host}:{port}")
print(f"📋 Environment: PORT={port_str}")

if __name__ == "__main__":
    import uvicorn
    # Railway 헬스체크를 위해 workers=1로 설정
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )

