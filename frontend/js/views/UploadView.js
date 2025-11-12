/**
 * UploadView - 파일 업로드 UI 관리
 */
export class UploadView {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container with id "${containerId}" not found`);
        }
        this.fileInput = null;
        this.analyzeButton = null;
        this.fileChangeCallback = null;
        this.analyzeClickCallback = null;
    }

    /**
     * 뷰 초기화
     */
    initialize() {
        this.fileInput = this.container.querySelector('#file-input');
        this.analyzeButton = this.container.querySelector('#analyze-btn');

        if (!this.fileInput || !this.analyzeButton) {
            throw new Error('Required elements not found in UploadView');
        }

        this.fileInput.addEventListener('change', (e) => {
            if (this.fileChangeCallback) {
                this.fileChangeCallback(e.target.files[0]);
            }
        });

        this.analyzeButton.addEventListener('click', () => {
            console.log('버튼 클릭 이벤트 발생, 콜백:', this.analyzeClickCallback);
            if (this.analyzeClickCallback) {
                this.analyzeClickCallback();
            } else {
                console.error('analyzeClickCallback이 설정되지 않았습니다!');
            }
        });
    }

    /**
     * 파일 변경 콜백 설정
     * @param {Function} callback - 파일 변경 시 호출될 콜백
     */
    onFileChange(callback) {
        this.fileChangeCallback = callback;
    }

    /**
     * 분석 버튼 클릭 콜백 설정
     * @param {Function} callback - 분석 버튼 클릭 시 호출될 콜백
     */
    onAnalyzeClick(callback) {
        this.analyzeClickCallback = callback;
    }

    /**
     * 분석 버튼 활성화/비활성화
     * @param {boolean} enabled - 활성화 여부
     */
    setAnalyzeButtonEnabled(enabled) {
        if (this.analyzeButton) {
            this.analyzeButton.disabled = !enabled;
            this.analyzeButton.textContent = enabled ? '영상 분석 시작' : '분석 중...';
        }
    }

    /**
     * 파일 입력 초기화
     */
    resetFileInput() {
        if (this.fileInput) {
            this.fileInput.value = '';
        }
    }
}

