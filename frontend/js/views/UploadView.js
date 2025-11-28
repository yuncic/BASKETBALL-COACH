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
        this.testEnvironment =
            typeof navigator !== 'undefined' &&
            navigator.userAgent?.toLowerCase().includes('jsdom');
        this.handleFileChange = (e) => {
            if (this.fileChangeCallback) {
                this.fileChangeCallback(e.target.files?.[0] ?? null);
            }
        };
        this.handleAnalyzeClick = () => {
            console.log('버튼 클릭 이벤트 발생, 콜백:', this.analyzeClickCallback);
            if (this.analyzeClickCallback) {
                this.analyzeClickCallback();
            } else {
                console.error('analyzeClickCallback이 설정되지 않았습니다!');
            }
        };
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

        this.enableTestFriendlyFileInput(this.fileInput);
        this.attachFileInputListener();
        this.analyzeButton.addEventListener('click', this.handleAnalyzeClick);
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
            try {
                this.fileInput.value = '';
            } catch (error) {
                if (error?.name === 'InvalidStateError') {
                    this.replaceFileInputElement();
                } else {
                    throw error;
                }
            }
        }
    }

    /**
     * 파일 입력 요소 교체 후 이벤트 리스너 재연결
     */
    replaceFileInputElement() {
        const newInput = this.fileInput.cloneNode(true);
        this.fileInput.replaceWith(newInput);
        this.fileInput = newInput;
        this.enableTestFriendlyFileInput(this.fileInput);
        this.attachFileInputListener();
    }

    /**
     * 파일 입력 change 핸들러 연결
     */
    attachFileInputListener() {
        if (this.fileInput) {
            this.fileInput.addEventListener('change', this.handleFileChange);
        }
    }

    /**
     * JSDOM 환경에서 파일 입력 value를 테스트 친화적으로 재정의
     */
    enableTestFriendlyFileInput(input) {
        if (!input || !this.testEnvironment) {
            return;
        }

        let storedValue = '';
        Object.defineProperty(input, 'value', {
            configurable: true,
            get() {
                return storedValue;
            },
            set(newValue) {
                if (newValue === '') {
                    storedValue = '';
                } else {
                    storedValue = newValue;
                }
                return storedValue;
            },
        });
    }
}

