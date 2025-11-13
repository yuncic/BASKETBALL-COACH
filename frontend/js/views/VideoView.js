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

        // result-section 표시 (container가 result-section이므로 직접 설정)
        this.container.style.display = 'flex';

        this.videoElement.src = videoURL;
        this.videoElement.load();
        this.downloadLink.href = downloadURL || videoURL;
        this.downloadLink.setAttribute('download', downloadName || 'result.mp4');
        this.downloadLink.setAttribute('type', 'video/mp4');
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

