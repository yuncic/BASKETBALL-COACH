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

        // result-section 표시 (container가 result-section이므로 직접 설정)
        this.container.style.display = 'flex';
        console.log('✅ result-section 표시됨:', this.container.style.display);

        // 기존 src 제거 후 새로 설정 (브라우저 캐시 문제 방지)
        this.videoElement.src = '';
        this.videoElement.load();
        
        // 짧은 딜레이 후 새 src 설정
        setTimeout(() => {
            this.videoElement.src = videoURL;
            this.videoElement.load();
            
            // 비디오 로드 이벤트 리스너 추가
            this.videoElement.onloadeddata = () => {
                console.log('✅ 비디오 데이터 로드 완료');
                this.videoElement.play().catch(e => {
                    console.warn('⚠️ 자동 재생 실패 (정상):', e);
                });
            };
            this.videoElement.onerror = (e) => {
                console.error('❌ 비디오 로드 에러:', e);
                console.error('비디오 요소 상태:', {
                    src: this.videoElement.src,
                    networkState: this.videoElement.networkState,
                    readyState: this.videoElement.readyState,
                    error: this.videoElement.error
                });
            };
            this.videoElement.oncanplay = () => {
                console.log('✅ 비디오 재생 준비 완료');
            };
        }, 100);
        
        this.downloadLink.href = downloadURL || videoURL;
        this.downloadLink.setAttribute('download', downloadName || 'result.mp4');
        this.downloadLink.setAttribute('type', 'video/mp4');
        
        console.log('✅ VideoView 설정 완료:', {
            videoSrc: this.videoElement.src,
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
    }

    /**
     * 비디오 초기화
     */
    reset() {
        this.hide();
    }
}

