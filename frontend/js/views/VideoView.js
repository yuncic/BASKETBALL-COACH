/**
 * VideoView - 비디오 재생 UI 관리
 */
export class VideoView {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container with id "${containerId}" not found`);
        }
        this.videoElement = null;
        this.downloadLink = null;
        this.currentVideoURL = null; // 중복 호출 방지용
    }

    /**
     * 뷰 초기화
     */
    initialize() {
        this.videoElement = this.container.querySelector('#result-video');
        this.downloadLink = this.container.querySelector('#download-link');

        if (!this.videoElement || !this.downloadLink) {
            throw new Error('Required elements not found in VideoView');
        }
    }

    /**
     * 비디오 표시
     * @param {string} videoURL - 비디오 URL
     * @param {string} downloadURL - 다운로드 URL
     */
    showVideo(videoURL, downloadURL, downloadName = 'result.mp4') {
        console.log('🎬 VideoView.showVideo 호출:', { videoURL, downloadURL, downloadName });
        
        if (!videoURL) {
            console.warn('⚠️ videoURL이 없습니다');
            this.hide();
            return;
        }

        // 중복 호출 방지: 같은 URL이면 스킵
        if (this.currentVideoURL === videoURL && this.videoElement && this.videoElement.src === videoURL) {
            console.log('⏭️ 같은 비디오 URL이므로 스킵:', videoURL);
            return;
        }
        this.currentVideoURL = videoURL;

        // result-section 표시 (container가 result-section이므로 직접 설정)
        this.container.style.display = 'flex';
        console.log('✅ result-section 표시됨:', this.container.style.display);

        // 비디오 요소 속성 확인
        if (!this.videoElement) {
            console.error('❌ videoElement가 없습니다!');
            this.videoElement = this.container.querySelector('#result-video');
            if (!this.videoElement) {
                console.error('❌ #result-video 요소를 찾을 수 없습니다!');
                return;
            }
        }

        // 기존 이벤트 리스너 제거
        this.videoElement.onloadeddata = null;
        this.videoElement.onerror = null;
        this.videoElement.oncanplay = null;

        // 비디오 로드 이벤트 리스너 추가 (src 설정 전에)
        this.videoElement.onloadedmetadata = () => {
            console.log('✅ 비디오 메타데이터 로드 완료');
            const videoWidth = this.videoElement.videoWidth;
            const videoHeight = this.videoElement.videoHeight;
            
            console.log('비디오 메타데이터:', {
                videoWidth: videoWidth,
                videoHeight: videoHeight,
                duration: this.videoElement.duration,
                naturalWidth: this.videoElement.naturalWidth,
                naturalHeight: this.videoElement.naturalHeight
            });
            
            // 모바일 회전 문제 해결: 비디오가 왼쪽으로 90도 회전되어 있다면 보정
            // 피드백 박스는 정상이므로 비디오 메타데이터의 회전 정보 때문
            const wrapper = this.container.querySelector('.video-wrapper');
            if (wrapper) {
                // 회전 초기화
                wrapper.style.transform = 'none';
                this.videoElement.style.transform = 'none';
                
                // 모바일에서 비디오가 왼쪽으로 90도 회전되어 보이는 경우
                // 세로 영상(높이 > 너비)이 가로로 표시되면 회전 보정 필요
                const isPortrait = videoHeight > videoWidth;
                const wrapperWidth = wrapper.offsetWidth || 270;
                const wrapperHeight = wrapper.offsetHeight || 480;
                
                console.log('비디오/컨테이너 크기 비교:', {
                    videoSize: `${videoWidth}x${videoHeight}`,
                    wrapperSize: `${wrapperWidth}x${wrapperHeight}`,
                    isPortrait: isPortrait
                });
                
                // 모바일에서 비디오가 왼쪽으로 90도 회전되어 보이는 경우
                // 피드백 박스는 정상이므로 비디오 메타데이터의 회전 정보 때문
                // 세로 영상인데 가로로 표시되면 회전 보정 필요
                
                // 사용자가 "왼쪽으로 90도 회전"이라고 했으므로, 오른쪽으로 90도 회전 보정
                // 또는 비디오가 실제로 회전되어 있다면 CSS로 보정
                
                // 세로 영상(높이 > 너비)이 가로 컨테이너에 들어가면 회전 보정
                if (isPortrait && wrapperWidth > wrapperHeight) {
                    console.log('⚠️ 비디오 회전 감지 - 오른쪽으로 90도 회전 보정');
                    this.videoElement.style.transform = 'rotate(90deg)';
                    this.videoElement.style.transformOrigin = 'center center';
                    // 회전 후 크기 조정
                    this.videoElement.style.width = '100%';
                    this.videoElement.style.height = 'auto';
                } else if (!isPortrait && wrapperHeight > wrapperWidth) {
                    console.log('⚠️ 비디오 회전 감지 - 왼쪽으로 90도 회전 보정');
                    this.videoElement.style.transform = 'rotate(-90deg)';
                    this.videoElement.style.transformOrigin = 'center center';
                    this.videoElement.style.width = '100%';
                    this.videoElement.style.height = 'auto';
                } else {
                    // 추가 확인: 비디오가 실제로 회전되어 있는지 확인
                    // 모바일에서 자주 발생하는 문제
                    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
                    if (isMobile && isPortrait) {
                        // 모바일에서 세로 영상이 회전되어 보이는 경우
                        console.log('⚠️ 모바일 세로 영상 회전 보정 시도');
                        // 일단 시도해보고 사용자가 확인
                        this.videoElement.style.transform = 'rotate(90deg)';
                        this.videoElement.style.transformOrigin = 'center center';
                    } else {
                        console.log('✅ 비디오 방향 정상 - 회전 불필요');
                    }
                }
            }
        };
        
        this.videoElement.onloadeddata = () => {
            console.log('✅ 비디오 데이터 로드 완료');
            this.videoElement.play().catch(e => {
                console.warn('⚠️ 자동 재생 실패 (정상):', e);
            });
        };
        this.videoElement.onerror = (e) => {
            console.error('❌ 비디오 로드 에러:', e);
            const error = this.videoElement.error;
            console.error('비디오 요소 상태:', {
                src: this.videoElement.src,
                networkState: this.videoElement.networkState,
                readyState: this.videoElement.readyState,
                error: error,
                errorCode: error ? error.code : null,
                errorMessage: error ? error.message : null,
                blobURL: videoURL,
                blobURLType: videoURL.startsWith('blob:') ? 'blob' : 'other'
            });
            
            // Blob URL이 유효한지 확인
            if (videoURL.startsWith('blob:')) {
                fetch(videoURL)
                    .then(response => {
                        console.log('Blob URL fetch 결과:', {
                            ok: response.ok,
                            status: response.status,
                            contentType: response.headers.get('content-type'),
                            size: response.headers.get('content-length')
                        });
                        return response.blob();
                    })
                    .then(blob => {
                        console.log('Blob 정보:', {
                            size: blob.size,
                            type: blob.type
                        });
                    })
                    .catch(err => {
                        console.error('Blob URL fetch 실패:', err);
                    });
            }
        };
        this.videoElement.oncanplay = () => {
            console.log('✅ 비디오 재생 준비 완료');
        };

        // 비디오 요소 속성 설정 (PC 브라우저 호환성 향상)
        this.videoElement.setAttribute('preload', 'auto');
        this.videoElement.setAttribute('crossOrigin', 'anonymous');
        this.videoElement.muted = true; // 자동 재생을 위해 음소거
        this.videoElement.playsInline = true; // 모바일 인라인 재생
        
        // 비디오 src 직접 설정
        this.videoElement.src = videoURL;
        
        // 비디오 요소가 보이도록 강제 (혹시 숨겨져 있을 수 있음)
        this.videoElement.style.display = 'block';
        this.videoElement.style.visibility = 'visible';
        this.videoElement.style.transform = 'none'; // 회전 초기화
        
        // 비디오 로드 시작
        this.videoElement.load();
        
        // PC 브라우저 호환성을 위해 약간의 딜레이 후 재시도
        setTimeout(() => {
            if (this.videoElement.readyState === 0 && this.videoElement.networkState === 3) {
                console.warn('⚠️ 비디오 로드 실패, 재시도 중...');
                this.videoElement.load();
            }
        }, 500);
        
        // video-container와 video-wrapper도 확인
        const videoContainer = this.container.querySelector('.video-container');
        const videoWrapper = this.container.querySelector('.video-wrapper');
        if (videoContainer) {
            videoContainer.style.display = 'flex';
        }
        if (videoWrapper) {
            videoWrapper.style.display = 'flex';
        }
        
        // 다운로드 링크 설정
        this.downloadLink.href = downloadURL || videoURL;
        this.downloadLink.setAttribute('download', downloadName || 'result.mp4');
        this.downloadLink.setAttribute('type', 'video/mp4');
        
        console.log('✅ VideoView 설정 완료:', {
            videoSrc: this.videoElement.src,
            videoElementExists: !!this.videoElement,
            videoElementDisplay: this.videoElement.style.display,
            containerDisplay: this.container.style.display,
            containerVisible: this.container.offsetParent !== null,
            downloadHref: this.downloadLink.href
        });
    }

    /**
     * 비디오 숨기기
     */
    hide() {
        if (this.container) {
            this.container.style.display = 'none';
        }
        if (this.videoElement) {
            this.videoElement.src = '';
        }
        this.currentVideoURL = null;
    }

    /**
     * 비디오 초기화
     */
    reset() {
        this.hide();
    }
}

