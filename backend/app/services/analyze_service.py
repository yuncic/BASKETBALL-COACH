import os
import math
import gc
# OpenCV headless 모드 설정 (GUI 라이브러리 불필요)
os.environ.setdefault('OPENCV_DISABLE_OPENCL', '1')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import cv2
import numpy as np
from pathlib import Path
from PIL import ImageFont, ImageDraw, Image

# PyTorch 2.6+ weights_only 문제 해결: torch.load를 패치
# 메모리 최적화 설정
try:
    import torch
    # 메모리 효율적인 설정
    torch.set_num_threads(1)  # CPU 스레드 제한
    if hasattr(torch, 'set_num_interop_threads'):
        torch.set_num_interop_threads(1)
    
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        # weights_only가 명시되지 않았거나 True인 경우 False로 변경
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        elif kwargs.get('weights_only') is True:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except Exception:
    pass

from ultralytics import YOLO

# PyTorch 2.1+ 보안 정책 대응: 모델 로드 전에 필요한 클래스들을 허용 목록에 추가
try:
    import torch
    # PyTorch 2.1+에서는 weights_only=True가 기본값이므로 필요한 클래스들을 추가
    if hasattr(torch.serialization, 'add_safe_globals'):
        import torch.nn as nn
        
        # PyTorch 기본 모듈들
        from torch.nn.modules.conv import Conv2d, Conv1d, Conv3d
        from torch.nn.modules.batchnorm import BatchNorm2d, BatchNorm1d, BatchNorm3d
        from torch.nn.modules.activation import ReLU, SiLU, LeakyReLU, Sigmoid, Tanh
        from torch.nn.modules.pooling import MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d
        from torch.nn.modules.linear import Linear
        from torch.nn.modules.dropout import Dropout, Dropout2d
        from torch.nn.modules.normalization import LayerNorm, GroupNorm
        from torch.nn.modules.container import ModuleList, ModuleDict, Sequential
        from torch.nn.modules.upsampling import Upsample
        
        # Ultralytics 모델 클래스들
        from ultralytics.nn.tasks import PoseModel, DetectionModel
        
        # Ultralytics 모듈 클래스들 - 동적으로 추가
        safe_globals_list = [
            # Python 내장 함수들 (PyTorch 모델 로드에 필요)
            getattr,
            setattr,
            # PyTorch 기본
            nn.Module,
            nn.Sequential,
            Sequential,  # container.Sequential
            ModuleList,  # container.ModuleList
            ModuleDict,  # container.ModuleDict
            Conv2d, Conv1d, Conv3d,
            BatchNorm2d, BatchNorm1d, BatchNorm3d,
            ReLU, SiLU, LeakyReLU, Sigmoid, Tanh,
            MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d,
            Linear,
            Dropout, Dropout2d,
            LayerNorm, GroupNorm,
            Upsample,  # upsampling.Upsample
            # Ultralytics 모델
            PoseModel,
            DetectionModel,
        ]
        
        # Ultralytics 모듈 클래스들을 동적으로 추가
        try:
            # ultralytics.nn.modules 패키지 import
            import ultralytics.nn.modules as ultralytics_modules
            
            # 먼저 conv 모듈에서 Conv를 가져와서 패키지 레벨에 alias
            try:
                from ultralytics.nn.modules.conv import Conv as ConvClass
                # 패키지 레벨에 Conv가 없으면 추가 (모델 파일이 ultralytics.nn.modules.Conv로 참조할 수 있음)
                if not hasattr(ultralytics_modules, 'Conv'):
                    setattr(ultralytics_modules, 'Conv', ConvClass)
                safe_globals_list.append(ConvClass)
            except Exception as e:
                print(f"⚠️ Conv 클래스 import 실패: {e}")
            
            # Concat도 동일하게 처리
            try:
                from ultralytics.nn.modules.block import Concat as ConcatClass
                if not hasattr(ultralytics_modules, 'Concat'):
                    setattr(ultralytics_modules, 'Concat', ConcatClass)
                safe_globals_list.append(ConcatClass)
            except:
                pass
            
            # 패키지 레벨의 모든 클래스 확인
            if hasattr(ultralytics_modules, 'Conv'):
                safe_globals_list.append(ultralytics_modules.Conv)
            if hasattr(ultralytics_modules, 'Concat'):
                safe_globals_list.append(ultralytics_modules.Concat)
            
            # 나머지 클래스들도 동적으로 추가
            for name in dir(ultralytics_modules):
                if not name.startswith('_') and name[0].isupper():
                    try:
                        obj = getattr(ultralytics_modules, name)
                        if isinstance(obj, type) and issubclass(obj, nn.Module):
                            if obj not in safe_globals_list:  # 중복 방지
                                safe_globals_list.append(obj)
                    except:
                        pass
        except Exception as e:
            print(f"⚠️ ultralytics.nn.modules import 실패: {e}")
            pass
        
        try:
            from ultralytics.nn.modules import conv as ultralytics_conv
            # Conv 클래스를 명시적으로 추가 (가장 중요!)
            if hasattr(ultralytics_conv, 'Conv'):
                safe_globals_list.append(ultralytics_conv.Conv)
            # conv 모듈의 모든 클래스 추가
            for name in dir(ultralytics_conv):
                if not name.startswith('_') and name[0].isupper():
                    try:
                        cls = getattr(ultralytics_conv, name)
                        if isinstance(cls, type) and issubclass(cls, nn.Module):
                            if cls not in safe_globals_list:  # 중복 방지
                                safe_globals_list.append(cls)
                    except:
                        pass
        except Exception as e:
            print(f"⚠️ ultralytics conv 모듈 import 실패: {e}")
            # 대안: 직접 Conv 클래스 import 시도
            try:
                from ultralytics.nn.modules.conv import Conv
                safe_globals_list.append(Conv)
            except:
                pass
        
        try:
            from ultralytics.nn.modules import block
            # block 모듈의 모든 클래스 추가
            for name in dir(block):
                if not name.startswith('_') and name[0].isupper():
                    try:
                        cls = getattr(block, name)
                        if isinstance(cls, type) and issubclass(cls, nn.Module):
                            safe_globals_list.append(cls)
                    except:
                        pass
        except:
            pass
        
        # head 모듈도 추가
        try:
            from ultralytics.nn.modules import head
            for name in dir(head):
                if not name.startswith('_') and name[0].isupper():
                    try:
                        cls = getattr(head, name)
                        if isinstance(cls, type) and issubclass(cls, nn.Module):
                            safe_globals_list.append(cls)
                    except:
                        pass
        except:
            pass
        
        torch.serialization.add_safe_globals(safe_globals_list)
except Exception:
    # PyTorch 2.0.x에서는 필요 없음, 또는 import 실패 시 무시
    pass

SLOW_FACTOR = 0.5
CONF_BALL = 0.20
SMOOTH_WIN = 5
DEFAULT_FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"


BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR
POSE_MODEL_PATH = MODEL_DIR / "yolov8n-pose.pt"
DET_MODEL_PATH = MODEL_DIR / "yolov8x.pt"  # 예전 코드와 동일: yolov8x 사용

def ensure_font(path):
    """폰트 경로를 확인하고, 실패 시 기본 폰트 반환"""
    try:
        ImageFont.truetype(path, 32)
        return path
    except:
        # 서버/맥 어디서든 동작하도록 폴백
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
    # Docker 컨테이너에서도 작동하도록 폰트 폴백 처리
    H, W = img.shape[:2]
    scale = H / 1920
    font_size = int(38 * scale)
    
    # 폰트 로드 시도 (여러 경로 시도)
    font = None
    font_paths_to_try = [
        ensure_font(font_path),  # 원래 경로 시도
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Noto CJK (한글 지원)
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # DejaVu (영문만)
    ]
    
    for path in font_paths_to_try:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except:
            continue
    
    # 모든 경로 실패 시 기본 폰트 사용
    if font is None:
        try:
            font = ImageFont.load_default()
        except:
            # 최후의 수단: 기본 폰트도 실패하면 None 사용 (텍스트는 표시되지만 폰트 없음)
            font = None
    
    img_pil = Image.fromarray(img)
    d = ImageDraw.Draw(img_pil)
    box = (int(40 * scale), int(40 * scale), int(1000 * scale), int((len(lines) + 1) * 60 * scale))
    d.rectangle(box, fill=(0, 0, 0, 180))
    y = int(70 * scale)
    for t in lines:
        d.text((int(60 * scale), y), t, fill=(255, 255, 255), font=font)
        y += int(60 * scale)
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
    
    # 모델 로드 전에 safe_globals가 확실히 설정되었는지 확인
    try:
        import torch
        import torch.nn as nn
        if hasattr(torch.serialization, 'add_safe_globals'):
            # 주요 클래스들을 다시 한 번 명시적으로 추가 (안전장치)
            additional_classes = []
            try:
                from ultralytics.nn.modules.block import C2f, C1, C2, C3, SPPF, Bottleneck
                additional_classes.extend([C2f, C1, C2, C3, SPPF, Bottleneck])
            except:
                pass
            try:
                import ultralytics.nn.modules as ultralytics_modules
                if hasattr(ultralytics_modules, 'Conv'):
                    additional_classes.append(ultralytics_modules.Conv)
                if hasattr(ultralytics_modules, 'Concat'):
                    additional_classes.append(ultralytics_modules.Concat)
            except Exception:
                pass
            
            if additional_classes:
                torch.serialization.add_safe_globals(additional_classes)
    except Exception:
        pass
    
    if _pose_model is None:
        # pose 모델 로드 직전에 모든 ultralytics 클래스를 확실히 추가
        try:
            import torch
            if hasattr(torch.serialization, 'add_safe_globals'):
                # Conv 클래스를 여러 경로에서 찾아서 추가
                conv_classes = []
                try:
                    from ultralytics.nn.modules.conv import Conv
                    conv_classes.append(Conv)
                except:
                    pass
                try:
                    import ultralytics.nn.modules as ultralytics_modules
                    # 패키지 레벨에 Conv가 있는지 확인
                    if hasattr(ultralytics_modules, 'Conv'):
                        conv_classes.append(ultralytics_modules.Conv)
                    # 없으면 conv 모듈에서 가져와서 추가
                    else:
                        try:
                            from ultralytics.nn.modules.conv import Conv as ConvClass
                            setattr(ultralytics_modules, 'Conv', ConvClass)
                            conv_classes.append(ConvClass)
                        except:
                            pass
                except:
                    pass
                
                # 모든 Conv 클래스를 추가
                if conv_classes:
                    torch.serialization.add_safe_globals(conv_classes)
                    print(f"✅ Conv 클래스 {len(conv_classes)}개 추가됨")
                
                # Concat도 추가
                try:
                    from ultralytics.nn.modules.block import Concat
                    torch.serialization.add_safe_globals([Concat])
                except:
                    pass
        except Exception as e:
            print(f"⚠️ safe_globals 추가 실패: {e}")
        _pose_model = YOLO(str(POSE_MODEL_PATH))
        # 메모리 최적화: 모델을 eval 모드로 설정하고 gradient 비활성화
        if hasattr(_pose_model, 'model'):
            _pose_model.model.eval()
            for param in _pose_model.model.parameters():
                param.requires_grad = False
        gc.collect()
    
    if _det_model is None:
        # detection 모델 로드 직전에 모든 ultralytics 클래스를 확실히 추가
        try:
            import torch
            if hasattr(torch.serialization, 'add_safe_globals'):
                # Conv 클래스를 여러 경로에서 찾아서 추가
                conv_classes = []
                try:
                    from ultralytics.nn.modules.conv import Conv
                    conv_classes.append(Conv)
                except:
                    pass
                try:
                    import ultralytics.nn.modules as ultralytics_modules
                    # 패키지 레벨에 Conv가 있는지 확인
                    if hasattr(ultralytics_modules, 'Conv'):
                        conv_classes.append(ultralytics_modules.Conv)
                    # 없으면 conv 모듈에서 가져와서 추가
                    else:
                        try:
                            from ultralytics.nn.modules.conv import Conv as ConvClass
                            setattr(ultralytics_modules, 'Conv', ConvClass)
                            conv_classes.append(ConvClass)
                        except:
                            pass
                except:
                    pass
                
                # 모든 Conv 클래스를 추가
                if conv_classes:
                    torch.serialization.add_safe_globals(conv_classes)
                    print(f"✅ Conv 클래스 {len(conv_classes)}개 추가됨")
                
                # Concat도 추가
                try:
                    from ultralytics.nn.modules.block import Concat
                    torch.serialization.add_safe_globals([Concat])
                except:
                    pass
        except Exception as e:
            print(f"⚠️ safe_globals 추가 실패: {e}")
        _det_model = YOLO(str(DET_MODEL_PATH))
        # 메모리 최적화: 모델을 eval 모드로 설정하고 gradient 비활성화
        if hasattr(_det_model, 'model'):
            _det_model.model.eval()
            for param in _det_model.model.parameters():
                param.requires_grad = False
        gc.collect()
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
    
    # 첫 프레임을 읽어서 실제 프레임 크기 확인
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("첫 프레임 읽기 실패")
    actual_frame_h, actual_frame_w = first_frame.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 첫 프레임으로 되돌리기
    
    print(f"📐 비디오 크기: 보고된 크기 {W}x{H}, 실제 프레임 {actual_frame_w}x{actual_frame_h}")
    
    # 모바일 세로 비디오 감지: 실제 프레임이 세로(H > W)이면 무조건 90도 시계방향 회전
    rotation_angle = 0
    if actual_frame_h > actual_frame_w:
        rotation_angle = 90
        print(f"📐 모바일 세로 비디오 감지 ({actual_frame_w}x{actual_frame_h}) → 90도 시계방향 회전")

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
        
        # 회전 메타데이터가 있으면 프레임 회전
        if rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
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

    # ---------- 타이밍 피크 탐색 ----------

    #시계열 기준 이전 동작보다 몇 초 앞인지 설정
    expected = {
        "elbow": -0.07,  
        "shoulder": -0.19,  
        "hip": -0.29,  
        "knee": -0.39, 
    }
    win_width = {"knee": 0.20, "hip": 0.20, "shoulder": 0.20, "elbow": 0.15}

    def pick_peak_in_window(t, signal, center_time, half_width):
        if (center_time is None) or (len(t) == 0):
            return None
        t0, t1 = center_time - half_width, center_time + half_width
        mask = (t >= t0) & (t <= t1)
        if not np.any(mask):
            return None
        s = signal.copy().astype(float)
        s[~mask] = np.nan
        z = zscore(s)
        if np.isfinite(z).sum() == 0:
            return None
        idx = int(np.nanargmax(z))
        return idx

    def fallback_peak(signal):
        z = zscore(signal)
        return int(np.nanargmax(z)) if np.isfinite(z).sum() > 0 else None

    knee_t = pick_peak_in_window(time, knee_v, REL + expected["knee"], win_width["knee"])
    hip_t = pick_peak_in_window(time, hip_v, REL + expected["hip"], win_width["hip"])
    sho_t = pick_peak_in_window(time, sho_v, REL + expected["shoulder"], win_width["shoulder"])
    elb_t = pick_peak_in_window(time, elb_v, REL + expected["elbow"], win_width["elbow"])

    if knee_t is None:
        knee_t = fallback_peak(knee_v)
    if hip_t is None:
        hip_t = fallback_peak(hip_v)
    if sho_t is None:
        sho_t = fallback_peak(sho_v)
    if elb_t is None:
        elb_t = fallback_peak(elb_v)

    def gap_time_by_index(idx_a, idx_b, fps_local):
        if (idx_a is None) or (idx_b is None):
            return None
        if idx_a < 0 or idx_b < 0:
            return None
        frame_gap = abs(idx_b - idx_a)
        return frame_gap / max(fps_local, 1e-6)

    # ---------- 타이밍 간격 ----------
    G_ke = None
    if (knee_t is not None) and (hip_t is not None):
        a, b = sorted([knee_t, hip_t])
        G_ke = gap_time_by_index(a, b, fps)

    G_sa = gap_time_by_index(sho_t, elb_t, fps)
    G_ar = gap_time_by_index(elb_t, release_idx, fps)

    # ---------- 점수/판정 ----------
    TARGET = {"G_ke": 0.0, "G_sa": 0.12, "G_ar": 0.07}
    TOL = {"G_ke": 0.05, "G_sa": 0.06, "G_ar": 0.05}

    def band_score(x, target, tol, max_penalty=60.0):
        if x is None or not np.isfinite(x):
            return 55.0
        diff = abs(x - target)
        if diff <= tol:
            return 100.0
        overshoot = diff - tol
        penalty = (overshoot / 0.30) * max_penalty
        return clamp01_100(100.0 - penalty)

    def verdict_sync_ke(x):
        if x is None or not np.isfinite(x):
            return "데이터 부족"
        diff = abs(x - 0.0)
        if 0 <= diff <= 0.03:
            return "완벽 동기화"
        elif 0.03 < diff <= 0.05:
            return "양호"
        elif 0.05 < diff <= 0.10:
            return "보통"
        elif 0.10 < diff <= 0.13:
            return "불량"
        elif 0.13 < diff <= 0.15:
            return "심각 불일치"
        else:
            return "판정 불가"

    def verdict_shoulder_elbow(x):
        if x is None or not np.isfinite(x):
            return "데이터 부족"
        if 0.00 < x < 0.20:
            return "빠름"
        elif 0.20 <= x <= 0.30:
            return "적절"
        elif 0.30 < x <= 0.340:
            return "느림"
        elif x > 0.50:
            return "매우 느림"
        else:
            return "판정 불가"

    def verdict_release(x):
        if x is None or not np.isfinite(x):
            return "데이터 부족"
        if 0.00 < x < 0.10:
            return "빠름"
        elif 0.10 <= x <= 0.20:
            return "적절"
        elif 0.20 < x <= 0.30:
            return "느림"
        elif x > 0.30:
            return "매우 느림"
        else:
            return "판정 불가"

    score_k = band_score(G_ke, TARGET["G_ke"], TOL["G_ke"])
    score_s = band_score(G_sa, TARGET["G_sa"], TOL["G_sa"])
    score_a = band_score(G_ar, TARGET["G_ar"], TOL["G_ar"])

    # ---------- 벡터 정렬 (팔·공, COM·공) ----------
    def regress_velocity(points, t, idx_center, pre=6, post=0):
        """idx_center 이전 구간을 이용한 최소자승 속도 추정(공/COM 공통)."""
        idxs = [i for i in range(idx_center - pre, idx_center + post + 1)
                if 0 <= i < len(points) and points[i] is not None]
        shrink = pre
        while len(idxs) < 2 and shrink > 1:  # 예전 코드와 동일: 최소 2개부터 시작
            shrink //= 2
            idxs = [i for i in range(idx_center - shrink, idx_center + 1)
                    if 0 <= i < len(points) and points[i] is not None]
        if len(idxs) < 2:  # 예전 코드와 동일: 2개 미만이면 None
            return None
        xs = np.array([points[i][0] for i in idxs], dtype=float)
        ys = np.array([points[i][1] for i in idxs], dtype=float)
        ts = np.array([t[i] for i in idxs], dtype=float)
        if len(idxs) < 3:  # 예전 코드와 동일: 3개 미만이면 None
            return None
        valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ts)
        if np.sum(valid) < 3:
            return None
        xs, ys, ts = xs[valid], ys[valid], ts[valid]
        A = np.vstack([ts, np.ones_like(ts)]).T
        ax, _ = np.linalg.lstsq(A, xs, rcond=None)[0]
        ay, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
        return np.array([ax, ay], dtype=float)

    def alignment_score(v1, v2):
        """코사인 유사도 기반 정렬도. 음수(반대방향)는 0 처리."""
        if v1 is None or v2 is None:
            return np.nan
        v1 = unit_vec(v1)
        v2 = unit_vec(v2)
        cosv = np.clip(np.dot(v1, v2), -1.0, 1.0)
        return clamp01_100(100.0 * max(0.0, cosv))

    ball_series = balls.copy()
    v_ball_img = regress_velocity(ball_series, time, release_idx, pre=6, post=0)
    v_ball = None if v_ball_img is None else np.array([v_ball_img[0], -v_ball_img[1]], dtype=float)

    kp_rel = kps[release_idx] if (0 <= release_idx < len(kps)) else None
    if kp_rel is None and len(kps) > 0:
        kp_rel = kps[max(0, release_idx - 1)]

    v_arm = None
    if kp_rel is not None:
        try:
            sh = kp_rel[R_SHO]
            wr = kp_rel[R_WRI]
            v_arm = np.array([wr[0] - sh[0], -(wr[1] - sh[1])], dtype=float)
        except:
            v_arm = None

    def com_points_from_kps(kps_list):
        pts = []
        for kp in kps_list:
            if kp is None:
                pts.append(None)
            else:
                try:
                    hip = kp[R_HIP]
                    sho = kp[R_SHO]
                    com = ((hip[0] + sho[0]) / 2.0, (hip[1] + sho[1]) / 2.0)
                    pts.append(com)
                except:
                    pts.append(None)
        return pts

    com_series = com_points_from_kps(kps)
    v_com_img = regress_velocity(com_series, time, release_idx, pre=6, post=0)
    v_com = None if v_com_img is None else np.array([v_com_img[0], -v_com_img[1]], dtype=float)

    score_arm = alignment_score(v_arm, v_ball)
    score_com = alignment_score(v_com, v_ball)

    def release_angle_horizontal(sh, wr):
        if (sh is None) or (wr is None):
            return np.nan
        dx = wr[0] - sh[0]
        dy_up = -(wr[1] - sh[1])
        ang = math.degrees(math.atan2(abs(dy_up), abs(dx)))
        if not np.isfinite(ang):
            return np.nan
        return float(max(0.0, min(90.0, ang)))

    rel_ang = np.nan
    if kp_rel is not None:
        try:
            rel_ang = release_angle_horizontal(kp_rel[R_SHO], kp_rel[R_WRI])
        except:
            rel_ang = np.nan

    timing_mean = np.nanmean([score_k, score_s, score_a])
    align_mean = np.nanmean([score_arm, score_com])
    eff_score = clamp01_100(0.5 * timing_mean + 0.5 * align_mean)

    # ---------- 패널 텍스트 ----------
    lines = [
        f"효율 점수: {eff_score:.1f}%",
        f"무릎↔허리 동기화: {fmt_sec(G_ke)} ({verdict_sync_ke(G_ke)})",
        f"어깨→팔꿈치: {fmt_sec(G_sa)} ({verdict_shoulder_elbow(G_sa)})",
        f"릴리즈 타이밍: {fmt_sec(G_ar)} ({verdict_release(G_ar)})",
        f"팔과 공의 방향 정렬도: {0.0 if not np.isfinite(score_arm) else score_arm:.1f}점",
        f"질량중심과 공의 방향 정렬도: {0.0 if not np.isfinite(score_com) else score_com:.1f}점",
        f"발사각: {rel_ang:.1f}°",
    ]

    # ---------- Pass2 렌더링 ----------
    cap = cv2.VideoCapture(input_path)
    
    # 회전 후 출력 크기 결정
    if rotation_angle in [90, 270]:
        output_width, output_height = H, W  # 가로/세로 교체
    else:
        output_width, output_height = W, H
    
    # Docker 환경 호환성을 위해 mp4v를 먼저 시도, 실패 시 XVID 폴백
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, max(fps * slow_factor, 1.0), (output_width, output_height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(output_path, fourcc, max(fps * slow_factor, 1.0), (output_width, output_height))
        if not out.isOpened():
            raise RuntimeError(f"VideoWriter 초기화 실패: mp4v와 XVID 모두 실패")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 회전 메타데이터가 있으면 프레임 회전
        if rotation_angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation_angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        pose_out = pose_model(frame)
        pose = pose_out[0]
        annotated = pose.plot()
        annotated = draw_panel(annotated, lines, font_path)
        out.write(annotated)

    cap.release()
    out.release()

    if (not os.path.exists(output_path)) or os.path.getsize(output_path) == 0:
        raise RuntimeError("주석 영상 생성 실패(파일이 비어있음). ffmpeg/코덱 점검 필요.")
    
    # PC 브라우저 호환성 및 모바일 회전 문제 해결을 위해 ffmpeg로 H.264 재인코딩
    temp_output = output_path + ".temp"
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        try:
            import subprocess
            print(f"🔄 ffmpeg 재인코딩 시작: {output_path} -> {temp_output}")
            
            # ffmpeg로 H.264 코덱으로 재인코딩
            # 프레임은 이미 회전되어 있으므로, 회전 메타데이터만 제거하면 됨
            # -map_metadata -1: 모든 메타데이터 제거
            # -metadata rotate=: 회전 메타데이터 명시적으로 제거
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", output_path,
                "-c:v", "libx264",  # H.264 코덱 (브라우저 호환성 최대)
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",  # 브라우저 호환성 필수
                "-movflags", "+faststart",  # 웹 스트리밍 최적화
                "-map_metadata", "-1",  # 모든 메타데이터 제거
                "-metadata", "rotate=",  # 회전 메타데이터 명시적으로 제거
                "-an",  # 오디오 제거
                "-f", "mp4",
                temp_output
            ]
            
            print(f"📋 ffmpeg 명령어: {' '.join(ffmpeg_cmd)}")
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0 and os.path.exists(temp_output) and os.path.getsize(temp_output) > 0:
                original_size = os.path.getsize(output_path)
                new_size = os.path.getsize(temp_output)
                os.replace(temp_output, output_path)
                print(f"✅ ffmpeg 재인코딩 완료: {original_size} bytes -> {new_size} bytes")
            else:
                print(f"⚠️ ffmpeg 재인코딩 실패 (원본 파일 사용)")
                print(f"   Return code: {result.returncode}")
                if result.stdout:
                    print(f"   stdout: {result.stdout[-1000:]}")
                if result.stderr:
                    print(f"   stderr: {result.stderr[-1000:]}")
                if os.path.exists(temp_output):
                    os.remove(temp_output)
        except FileNotFoundError:
            print(f"⚠️ ffmpeg가 설치되지 않음 (원본 파일 사용)")
            if os.path.exists(temp_output):
                os.remove(temp_output)
        except subprocess.TimeoutExpired:
            print(f"⚠️ ffmpeg 재인코딩 타임아웃 (원본 파일 사용)")
            if os.path.exists(temp_output):
                os.remove(temp_output)
        except Exception as e:
            print(f"⚠️ ffmpeg 재인코딩 중 오류 (원본 파일 사용): {e}")
            if os.path.exists(temp_output):
                os.remove(temp_output)

    # ---------- 웹 패널용 리포트(영상 안 패널과 동일 정보) ----------
    report = {
        "eff_score": round(float(eff_score), 1),
        "metrics": {
            "knee_hip": {
                "gap": fmt_sec((abs(knee_t - hip_t) / fps) if (knee_t is not None and hip_t is not None) else None),
                "verdict": verdict_sync_ke(G_ke),
            },
            "shoulder_elbow": {
                "gap": fmt_sec(G_sa),
                "verdict": verdict_shoulder_elbow(G_sa),
            },
            "release_timing": {
                "gap": fmt_sec(G_ar),
                "verdict": verdict_release(G_ar),
            },
        },
        "alignment": {
            "arm_ball": 0.0 if not np.isfinite(score_arm) else round(float(score_arm), 1),
            "com_ball": 0.0 if not np.isfinite(score_com) else round(float(score_com), 1),
            "release_angle": 0.0 if not np.isfinite(rel_ang) else round(float(rel_ang), 1),
        },
        "suggestions": [],
    }

    # 간단 피드백(원본 흐름 유지)
    if eff_score < 60:
        report["suggestions"].append("하체 리듬과 릴리즈 타이밍의 일관성을 높이면 슛 효율이 향상됩니다.")
    elif eff_score < 80:
        report["suggestions"].append("팔꿈치와 손목의 타이밍을 조정해 릴리즈를 더 부드럽게 만들어보세요.")
    else:
        report["suggestions"].append("좋은 폼입니다! 릴리즈 타이밍만 유지하면 안정적인 슛이 가능합니다.")

    rv = report["metrics"]["release_timing"]["verdict"]
    if rv == "느림":
        report["suggestions"].append("릴리즈가 느립니다. 하체 힘 전달 직후 릴리즈 타이밍을 앞당겨보세요.")
    elif rv == "빠름":
        report["suggestions"].append("릴리즈가 빠릅니다. 하체-상체 순차 힘 전달 후에 릴리즈하세요.")

    return report
    

    
