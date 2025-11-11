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

# COCO right side indices
R_SHO, R_ELB, R_WRI, R_HIP, R_KNE, R_ANK = 6, 8, 10, 12, 14, 16

def analyze_video_from_path(
    input_path: str,
    output_path: str,
    font_path: str = DEFAULT_FONT,
    slow_factor: float = SLOW_FACTOR
):
    """
    '원본 분석 로직'을 그대로 보존한 형태로 함수화.
    - input_path: 입력 영상 경로(.mp4, .mov 상관없음)
    - output_path: 결과 주석 영상 저장 경로(mp4)
    - font_path: 패널 폰트 경로
    - slow_factor: 재생 속도 배수(0.5면 절반 속도)

    반환: report(dict) - 웹에서 오른쪽 패널에 텍스트로 표시
    """
    pose_model, det_model = _get_models()

    # ---------- Pass1: 포즈 & 공 궤적 ----------
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {input_path}")

    fps_reported = cap.get(cv2.CAP_PROP_FPS) or 0.0
    fps = fps_reported if (10.0 <= fps_reported <= 240.0) else 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    time = []  # 초 단위
    knees = []
    hips = []
    shoulders = []
    elbows = []
    wrists = []
    balls = []
    kps = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        time.append(t_ms / 1000.0 if (t_ms and t_ms > 0) else (len(time) / fps))

        pose_out = pose_model(frame)
        pose = pose_out[0]
        kp = None
        if (pose.keypoints is not None) and hasattr(pose.keypoints, "xy") and len(pose.keypoints.xy) > 0:
            kp = pose.keypoints.xy[0].cpu().numpy()

        det = det_model(frame)[0]
        bxy = None
        if det and det.boxes is not None and len(det.boxes) > 0:
            best_conf = -1.0
            for xyxy, c, conf in zip(det.boxes.xyxy, det.boxes.cls, det.boxes.conf):
                try:
                    name = det_model.names[int(c)].lower()
                except:
                    name = ""
                if ("ball" in name) and float(conf) >= CONF_BALL:
                    x1, y1, x2, y2 = xyxy.cpu().numpy()
                    if float(conf) > best_conf:
                        best_conf = float(conf)
                        bxy = ((x1 + x2) / 2, (y1 + y2) / 2)

        if kp is None:
            knees.append(np.nan)
            hips.append(np.nan)
            shoulders.append(np.nan)
            elbows.append(np.nan)
            wrists.append(None)
            balls.append(bxy)
            kps.append(None)
            continue

        an, k, h = kp[R_ANK], kp[R_KNE], kp[R_HIP]
        sh, el, wr = kp[R_SHO], kp[R_ELB], kp[R_WRI]

        knees.append(angle_abc(an, k, h))  # 무릎 폄 증가
        hips.append(angle_abc(k, h, sh))  # 허리 폄 근사
        shoulders.append(angle_abc(h, sh, el))  # 어깨 굴곡 근사
        elbows.append(angle_abc(sh, el, wr))  # 팔꿈치 폄 증가
        wrists.append(tuple(wr) if wr is not None else None)
        balls.append(bxy)
        kps.append(kp)

    cap.release()
    time = np.asarray(time, float)
    nT = len(time)
    
    # ---------- 시계열 평활화 & 각속도 ----------
    knees_s = smooth(knees)
    hips_s = smooth(hips)
    shoulders_s = smooth(shoulders)
    elbows_s = smooth(elbows)
    knee_v = derivative(knees_s, time)
    hip_v = derivative(hips_s, time)
    sho_v = derivative(shoulders_s, time)
    elb_v = derivative(elbows_s, time)

    # ---------- 릴리즈 검출 (손목 제외 / 공 궤도 기반) ----------
    ball_y = np.array([b[1] if b is not None else np.nan for b in balls], dtype=float)
    ball_y_s = smooth(ball_y)
    v_ball_y = derivative(ball_y_s, time)

    release_idx = None
    for i in range(1, nT):
        if (balls[i - 1] is not None) and (balls[i] is None):
            release_idx = i
            break

    if release_idx is None:
        signs = np.sign(v_ball_y)
        change_pts = np.where(np.diff(signs) > 0)[0]
        if len(change_pts) > 0:
            release_idx = int(change_pts[0] + 1)

    if release_idx is None:
        dist_speed = np.gradient(np.abs(v_ball_y))
        if np.nanmax(dist_speed) > np.nanmean(dist_speed) * 3:
            release_idx = int(np.nanargmax(dist_speed))

    if release_idx is None:
        release_idx = int(nT * 0.7) if nT > 0 else 0

    REL = time[release_idx] if nT > 0 else 0.0


def analyze_video_from_path(
    input_path: str,
    output_path: str,
    font_path: str = None,
    slow_factor: float = None
):
    

    pass