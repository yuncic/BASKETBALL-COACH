# 🏀 Shooting Analyzer

## 프로그램 개요

* 처음 개발공부를 시작할떄부터 만들고 싶었던 앱이었음.
* 농구부를 좋아하고, 농구부 활동을 하면서 슈팅의 효율도를 분석해보고 싶었음.
* 슛은 상당히 주관적이기 때문에 정답이 없음 -> 객관적인 역학적 지표를 사용하여 힘의 효율도를 기준으로 분석했음.
* 스포츠 전공자이기 때문에 전공과 관련된 프로젝트를 해보고 싶었음.
* 나에게 상대적으로 익숙한 python으로 모델을 구축하고, 프리코스 기간에 익힌 JS로 프론트를 구현.
* 프리코스 기간에 배운 MVC 패턴을 적용하고, 프론트 부분은 TDD 형식으로 구현.
* 실제로 배포까지 도전!! -> 8기의 슬로건인 "도전"에 걸맞게 항상 배포에 대한 두려움을 극복.

## 구현할 기능

### 백엔드

### FastAPI 앱 및 라우터 설정
* main.py에서 FastAPI 인스턴스를 생성하고 CORS·정적 파일 서빙 설정
* /api/analyze, /api/report, /health 라우트를 등록해 분석 요청·리포트 조회·헬스 체크 처리

### 영상 업로드 처리 및 파일 검증
* analyze.py에서 UploadFile을 통해 multipart/form-data 요청 수신
* 확장자/용량 등을 검증하면서 임시 파일로 저장

### YOLO 기반 분석 파이프라인 실행
* 포즈 추정·공 탐지 변환 (Ultralytics YOLOv8)
* NumPy로 평활화·파생량 계산, 타이밍/정렬도 지표 산출

### 주석 영상 생성 및 리포트 작성
* OpenCV + Pillow로 분석 결과를 오버레이한 영상 생성
* 효율 점수·판정·개선 제안이 포함된 JSON 리포트 구성

### API 응답 및 정적 파일 서빙
* 분석 결과 영상은 바이너리 스트림으로 반환, 리포트는 Base64/경로 함께 제공
* 프론트 정적 자산(HTML·JS·CSS)을 FastAPI에서 직접 제공

### 보조 관리 기능
* YOLO 모델 자동 다운로드 스크립트 제공

### 프론트엔드

### 촬영 가이드 및 상태 안내 UI
* 90도 측면, 전신 촬영, 배경 조건 등 촬영 가이드 카드 표시
* 분석 진행 시 30초~1분 소요 메시지를 출력하고 완료 후 자동 숨김

### 영상 업로드 및 분석 요청 흐름
* 파일 선택 시 검증(비디오 형식, 중복 초기화)과 버튼 활성화
* 분석 버튼을 누르면 폼 데이터를 백엔드 /api/analyze로 전송

### 분석 결과 표시
* 반환받은 Blob으로 결과 영상을 재생하고 *.mp4로 다운로드 가능
* JSON 리포트를 파싱해 효율 점수, 타이밍 판정, 무게중심-공 정렬도, 개선 제안 등을 렌더링

### MVC 구조 + TDD 적용
* Model(상태 관리), View(UI 렌더링), Controller(이벤트/흐름 제어) 분리
* Jest + jsdom으로 Model·View·Controller·Service 계층 테스트

### 반응형 레이아웃 및 예외 처리
* 모바일·데스크톱 환경에서 UI가 자연스럽게 배치되도록 CSS 구성
* 분석 실패 시 오류 메시지 표시 및 상태 초기화


## 사용된 기술/라이브러리

### Backend

* fastAPI: Rest API 서버 구현, 농구 슛 영상 파일 업로드 처리
선택 이유: Flask 나 Django 보다 사용법이 편리하고, 나에게 가장 익숙하기 떄문

* python-multipart: 파일 업로드(Form Data 지원)
선택 이유: fastAPI에서 UploadFile을 쓰기 위해 필수임

* CORSMiddleware: CORS(Cross-Origin Resource Sharing) 제약을 완화하는 역할

* FileResponse: FastAPI가 파일을 HTTP 응답으로 직접 반환할 수 있게 도와주는 응답 클래스

* pathlib: 운영체제별 경로 차이(슬래시 방향 등)를 신경 쓰지 않고 안전하게 파일 위치를 찾기 위한 도구

### 영상 분석 모델

* OpenCV: 영상 프레임 처리, 주석 영상 출력
선택 이유: 프레임 단위 처리와 영상 저장에 검증된 라이브러리이기 때문

* Ultralytics YOLOv8: 포즈 추정, 농구공 탐지 (실시간 객체 탐지를 위한 강력한 딥러닝 알고리즘)
선택 이유: 최신 YOLO 모델을 간단히 사용할 수 있는 고수준 인터페이스를 제공하고, 고정밀 포즈/객체 인식에 적합하기 떄문

* numpy: 수치 계산

* PIL: 영상 위 텍스트 패널 렌더링
선택 이유: OpenCV만으로는 텍스트 렌더링이 번거로워 PIL(Python Image Library)를 보조로 사용

### Frontend
* JavaScript: MVC 구조, TDD 기반 기능 구현
* HTML/CSS: UI 구성, 레이아웃
* Jest: 프론트 테스트

## 학습 및 도전 포인트
* MVC 패턴 개발 착수
* TDD 방식 프론트 구현
* 로컬에서 뿐만이 아니라 배포까지해서 실유저 경험


## 프로젝트 구조

```
Shooting-Analyzer/
├── backend/                 # 백엔드 (FastAPI)
│   ├── app/
│   │   ├── main.py         # FastAPI 서버 진입점
│   │   ├── routes/         # API 라우트 (컨트롤러 역할)
│   │   ├── services/       # 비즈니스 로직 (서비스 레이어)
│   │   └── models/         # 데이터 모델
│   ├── requirements.txt    # 백엔드가 실행되는 데 필요한 파이썬 패키지 목록을 정리해 둔 파일
│
├── frontend/                # 프론트엔드 (순수 JS, MVC 패턴)
│   ├── index.html          # 메인 HTML
│   ├── css/
│   │   └── style.css       # 스타일시트
│   ├── js/
│   │   ├── models/         # Model: 데이터 관리
│   │   ├── views/          # View: UI 렌더링
│   │   ├── controllers/    # Controller: 이벤트 처리
│   │   ├── services/       # API 통신
│   │   └── app.js          # 앱 진입점
│   └── tests/              # 프론트엔드 테스트 (TDD)
│
└── README.md
```


