# Railway 배포 가이드

## Railway 배포 방법

### 1. Railway 계정 생성
- https://railway.app 접속
- GitHub 계정으로 로그인

### 2. 새 프로젝트 생성
1. "New Project" 클릭
2. "Deploy from GitHub repo" 선택
3. 저장소 선택

### 3. 환경 변수 설정 (선택사항)
- Railway 대시보드에서 "Variables" 탭
- 필요시 환경 변수 추가

### 4. 배포
- Railway가 자동으로 Dockerfile을 감지하고 빌드/배포
- 빌드 로그에서 진행 상황 확인

### 5. 도메인 설정
- "Settings" → "Generate Domain" 클릭
- 자동으로 HTTPS 도메인 생성

## Dockerfile 장점

1. **Python 버전 완전 제어**: Python 3.10 고정
2. **의존성 관리**: requirements.txt로 명확한 버전 관리
3. **환경 일관성**: 로컬과 동일한 환경 보장
4. **빌드 캐싱**: 레이어별 캐싱으로 빠른 빌드

## Railway vs Render

| 항목 | Railway | Render |
|------|---------|--------|
| Docker 지원 | ✅ 완벽 지원 | ⚠️ 제한적 |
| Python 버전 제어 | ✅ Dockerfile로 완전 제어 | ⚠️ runtime.txt (때로 무시됨) |
| 빌드 속도 | 빠름 | 보통 |
| 무료 티어 | $5 크레딧/월 | 750시간/월 |
| 설정 복잡도 | 낮음 | 낮음 |

## 문제 해결

### 빌드 실패 시
1. Railway 대시보드에서 빌드 로그 확인
2. Dockerfile의 Python 버전 확인
3. requirements.txt 의존성 확인

### 메모리 부족 시
- Railway 대시보드에서 리소스 업그레이드
- 또는 모델 파일을 빌드 시점에 다운로드하도록 Dockerfile 수정

