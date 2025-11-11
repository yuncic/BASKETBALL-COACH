import os
import math
import cv2
import numpy as np
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO

SLOW_FACTOR = 0.5
CONF_BALL = 0.20
SMOOTH_WIN = 5
DEFAULT_FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"


BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR
POSE_MODEL_PATH = MODEL_DIR / "yolov8n-pose.pt"
DET_MODEL_PATH = MODEL_DIR / "yolov8x.pt"

def ensure_font(path):
    try:
        ImageFont.truetype(path, 32)
        return path
    except:
        return DEFAULT_FONT

def angle_abc(a, b, c):
    '''<ABC (B가 꼭짓점) 를 degree로 계산. 누락 시 None 반환.'''
    if a is None or b is None or c is None:
        return None
    ax, ay = a
    bx, by = b
    cx, cy = c
    AB = (ax - bx, ay - by) # 꼭짓점 B에서 A로 향하는 벡터
    CB = (cx - bx, cy - by) # 꼭짓점 B에서 C로 향하는 벡터
    #백터길이
    dab = math.hypot(*AB) 
    dcb = math.hypot(*CB)
    if dab < 1e-6 or dcb < 1e-6:
        return None
    cosv = (AB[0] * CB[0] + AB[1] * CB[1]) / (dab * dcb + 1e-6)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))


def smooth(x, win=SMOOTH_WIN, repeat=3):
    """이동평균 기반 평활화 + NaN 보간."""
    arr = np.array(x, dtype=float)
    if np.isnan(arr).any():
        n = len(arr)
        idx = np.arange(n)
        mask = ~np.isnan(arr)
        arr = np.interp(idx, idx[mask], arr[mask]) if mask.any() else np.zeros_like(arr)
    win = max(1, int(win))
    ker = np.ones(win) / win
    for _ in range(repeat):
        arr = np.convolve(arr, ker, mode="same")
    return arr


def derivative(y, t):
    """시간축 기준 1차 미분(중앙차분)."""
    y = np.asarray(y, float)
    t = np.asarray(t, float)
    dy = np.full_like(y, np.nan, dtype=float)
    n = len(y)
    if n < 2:
        return np.zeros_like(y, dtype=float)
    for i in range(n):
        if i == 0:
            dt = t[1] - t[0]
            dy[i] = (y[1] - y[0]) / (dt + 1e-6)
        elif i == n - 1:
            dt = t[-1] - t[-2]
            dy[i] = (y[-1] - y[-2]) / (dt + 1e-6)
        else:
            dt = t[i + 1] - t[i - 1]
            dy[i] = (y[i + 1] - y[i - 1]) / (dt + 1e-6)
    return dy


def zscore(x):
    x = np.asarray(x, float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-8:
        return np.zeros_like(x)
    return (x - m) / s


def clamp01_100(x):
    return max(0.0, min(100.0, float(x)))


def fmt_sec(x):
    return f"{x:.2f}s" if (x is not None and np.isfinite(x)) else "-"

def draw_panel(img, lines, font_path):
    H, W = img.shape[:2]
    scale = H / 1920 #영상 높이 1920을 기준으로 scale factor 생성
    font = ImageFont.truetype(ensure_font(font_path), int(38 * scale))
    img_pil = Image.fromarray(img)
    d = ImageDraw.Draw(img_pil)
    box = (int(40 * scale), int(40 * scale), int(1000 * scale), int((len(lines) + 1) * 60 * scale))
    d.rectangle(box, fill=(0, 0, 0, 180))
    y = int(70 * scale)
    for t in lines:
        d.text((int(60 * scale), y), t, fill=(255, 255, 255), font=font)
        y += int(60 * scale) #줄바꿈
    return np.array(img_pil)

def unit_vec(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else np.zeros_like(v)

_pose_model = None
_det_model = None

def _get_models():
    """모델을 지연 로드 (첫 호출 시 로드)"""
    global _pose_model, _det_model
    if _pose_model is None:
        _pose_model = YOLO(str(POSE_MODEL_PATH))
    if _det_model is None:
        _det_model = YOLO(str(DET_MODEL_PATH))
    return _pose_model, _det_model

def analyze_video_from_path(
    input_path: str,
    output_path: str,
    font_path: str = None,
    slow_factor: float = None
):
    

    pass