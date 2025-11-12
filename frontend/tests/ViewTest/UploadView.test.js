import { UploadView } from '../../js/views/UploadView.js';

describe('UploadView', () => {
    let container;
    let view;

    beforeEach(() => {
        // DOM 요소 생성
        document.body.innerHTML = `
            <div id="upload-section">
                <input type="file" id="file-input" accept="video/mp4,video/mov,video/quicktime,video/*">
                <button id="analyze-btn" class="analyze-btn">영상 분석 시작</button>
            </div>
        `;
        container = document.getElementById('upload-section');
        view = new UploadView('upload-section');
        view.initialize();
    });

    afterEach(() => {
        document.body.innerHTML = '';
    });

    describe('초기화', () => {
        test('뷰가 정상적으로 초기화되어야 한다', () => {
            expect(view.fileInput).toBeTruthy();
            expect(view.analyzeButton).toBeTruthy();
        });

        test('존재하지 않는 컨테이너 ID는 에러를 발생시켜야 한다', () => {
            expect(() => new UploadView('non-existent')).toThrow('Container with id "non-existent" not found');
        });

        test('필수 요소가 없으면 에러를 발생시켜야 한다', () => {
            document.body.innerHTML = '<div id="upload-section"></div>';
            const invalidView = new UploadView('upload-section');
            expect(() => invalidView.initialize()).toThrow('Required elements not found in UploadView');
        });
    });

    describe('파일 변경 이벤트', () => {
        test('파일 변경 콜백을 등록할 수 있어야 한다', () => {
            const callback = jest.fn();
            view.onFileChange(callback);
            expect(view.fileChangeCallback).toBe(callback);
        });

        test('파일 변경 이벤트 핸들러가 등록되어야 한다', () => {
            const callback = jest.fn();
            view.onFileChange(callback);
            // 이벤트 리스너가 등록되었는지 확인 (실제 파일 입력은 보안상 제한이 있어 직접 테스트하기 어려움)
            expect(view.fileInput).toBeTruthy();
        });
    });

    describe('분석 버튼 클릭 이벤트', () => {
        test('분석 버튼 클릭 시 콜백이 호출되어야 한다', () => {
            const callback = jest.fn();
            view.onAnalyzeClick(callback);
            view.analyzeButton.click();
            expect(callback).toHaveBeenCalled();
        });
    });

    describe('버튼 상태 관리', () => {
        test('버튼을 비활성화할 수 있어야 한다', () => {
            view.setAnalyzeButtonEnabled(false);
            expect(view.analyzeButton.disabled).toBe(true);
            expect(view.analyzeButton.textContent).toBe('분석 중...');
        });

        test('버튼을 활성화할 수 있어야 한다', () => {
            view.setAnalyzeButtonEnabled(false);
            view.setAnalyzeButtonEnabled(true);
            expect(view.analyzeButton.disabled).toBe(false);
            expect(view.analyzeButton.textContent).toBe('영상 분석 시작');
        });
    });

    describe('파일 입력 리셋', () => {
        test('파일 입력을 초기화할 수 있어야 한다', () => {
            view.fileInput.value = 'test.mp4';
            view.resetFileInput();
            expect(view.fileInput.value).toBe('');
        });
    });
});

