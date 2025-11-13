#!/usr/bin/env python3
"""Railway 배포용 시작 스크립트"""
import os
import sys

port = int(os.environ.get("PORT", "10000"))
host = "0.0.0.0"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=host, port=port)

