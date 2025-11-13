/**
 * VideoModel - 비디오 파일 및 상태 관리
 */
export class VideoModel {
    constructor() {
        this.file = null;
        this.videoURL = null;
        this.downloadLink = null;
        this.downloadFilename = 'result.mp4';
        this.listeners = [];
    }

    /**
     * 파일 설정
     * @param {File} file - 비디오 파일
     */
    setFile(file) {
        if (file && !this.isValidVideoFile(file)) {
            throw new Error("지원하지 않는 비디오 파일 형식입니다.");
        }
        this.file = file;
        this.videoURL = null;
        this.downloadLink = null;
        this.notifyListeners();
    }

    /**
     * 비디오 URL 설정
     * @param {string} url - 비디오 URL
     */
    setVideoURL(url) {
        this.videoURL = url;
        this.notifyListeners();
    }

    /**
     * 다운로드 링크 설정
     * @param {string} url - 다운로드용 URL
     */
    setDownloadLink(url) {
        this.downloadLink = url;
        this.notifyListeners();
    }

    /**
     * 비디오 상태 초기화
     */
    reset() {
        this.file = null;
        this.videoURL = null;
        this.downloadLink = null;
        this.downloadFilename = 'result.mp4';
        this.notifyListeners();
    }

    /**
     * 비디오 파일 유효성 검사
     * @param {File} file - 검사할 파일
     * @returns {boolean} 유효한 비디오 파일인지 여부
     */
    isValidVideoFile(file) {
        const validTypes = [
            'video/mp4',
            'video/mov',
            'video/quicktime',
            'video/webm',
            'video/avi'
        ];
        return validTypes.includes(file.type) || file.name.match(/\.(mp4|mov|webm|avi)$/i);
    }

    /**
     * 파일 가져오기
     * @returns {File|null} 현재 파일
     */
    getFile() {
        return this.file;
    }

    /**
     * 비디오 URL 가져오기
     * @returns {string|null} 현재 비디오 URL
     */
    getVideoURL() {
        return this.videoURL;
    }

    /**
     * 다운로드 링크 가져오기
     * @returns {string|null} 현재 다운로드 링크
     */
    getDownloadLink() {
        return this.downloadLink;
    }

    /**
     * 다운로드 파일명 설정
     * @param {string} name - 파일명
     */
    setDownloadFilename(name) {
        this.downloadFilename = name || 'result.mp4';
        this.notifyListeners();
    }

    /**
     * 다운로드 파일명 반환
     * @returns {string}
     */
    getDownloadFilename() {
        return this.downloadFilename;
    }

    /**
     * 변경 리스너 등록
     * @param {Function} callback - 변경 시 호출될 콜백
     */
    subscribe(callback) {
        this.listeners.push(callback);
    }

    /**
     * 변경 리스너 제거
     * @param {Function} callback - 제거할 콜백
     */
    unsubscribe(callback) {
        this.listeners = this.listeners.filter(listener => listener !== callback);
    }

    /**
     * 모든 리스너에 변경 알림
     */
    notifyListeners() {
        this.listeners.forEach(callback => {
            try {
                callback(this);
            } catch (error) {
                console.error('VideoModel listener error:', error);
            }
        });
    }
}

