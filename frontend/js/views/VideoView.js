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
        if (!videoURL) {
            this.hide();
            return;
        }

        this.videoElement.src = videoURL;
        this.videoElement.load();
        
        // 모바일 다운로드 문제 해결: blob URL을 직접 사용하거나 새 창에서 열기
        if (downloadURL && downloadURL.startsWith('blob:')) {
            // 모바일에서는 blob URL 다운로드가 제한될 수 있으므로 클릭 이벤트로 처리
            this.downloadLink.href = downloadURL;
            this.downloadLink.setAttribute('download', downloadName || 'result.mp4');
            this.downloadLink.setAttribute('type', 'video/mp4');
            
            // 모바일에서 다운로드가 안 될 경우를 대비해 클릭 이벤트 추가
            const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
            if (isMobile) {
                // 기존 클릭 이벤트 제거 후 새로 추가
                this.downloadLink.onclick = (e) => {
                    // 모바일에서 blob URL 다운로드가 안 될 경우 새 창에서 열기
                    const link = document.createElement('a');
                    link.href = downloadURL;
                    link.download = downloadName || 'result.mp4';
                    link.target = '_blank';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                };
            }
        } else {
            this.downloadLink.href = downloadURL || videoURL;
            this.downloadLink.setAttribute('download', downloadName || 'result.mp4');
            this.downloadLink.setAttribute('type', 'video/mp4');
        }
        
        this.container.style.display = 'flex';
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

