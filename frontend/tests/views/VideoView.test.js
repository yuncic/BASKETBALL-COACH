import { VideoView } from '../../js/views/VideoView.js';

describe('VideoView', () => {
    let container;
    let view;

    beforeEach(() => {
        // DOM 요소 생성
        document.body.innerHTML = `
            <div id="result-section" class="result-section" style="display: none;">
                <div class="video-container">
                    <div class="video-wrapper">
                        <video id="result-video" controls autoplay muted playsinline></video>
                    </div>
                    <a id="download-link" href="#" download="result.mp4" class="download-link">
                        🎥 분석 결과 영상 다운로드
                    </a>
                </div>
            </div>
        `;
        container = document.getElementById('result-section');
        view = new VideoView('result-section');
        view.initialize();
    });

    afterEach(() => {
        document.body.innerHTML = '';
    });

    describe('초기화', () => {
        test('뷰가 정상적으로 초기화되어야 한다', () => {
            expect(view.videoElement).toBeTruthy();
            expect(view.downloadLink).toBeTruthy();
            expect(view.container).toBe(container);
        });

        test('존재하지 않는 컨테이너 ID는 에러를 발생시켜야 한다', () => {
            expect(() => new VideoView('non-existent')).toThrow('Container with id "non-existent" not found');
        });

        test('필수 요소가 없으면 에러를 발생시켜야 한다', () => {
            document.body.innerHTML = '<div id="result-section"></div>';
            const invalidView = new VideoView('result-section');
            expect(() => invalidView.initialize()).toThrow('Required elements not found in VideoView');
        });
    });

    describe('비디오 표시', () => {
        test('비디오 URL을 설정할 수 있어야 한다', () => {
            const videoURL = 'http://example.com/video.mp4';
            const downloadURL = 'http://example.com/download.mp4';
            view.showVideo(videoURL, downloadURL, 'custom.mp4');
            expect(view.videoElement.src).toContain(videoURL);
            expect(view.downloadLink.href).toBe(downloadURL);
            expect(view.downloadLink.getAttribute('download')).toBe('custom.mp4');
            expect(view.downloadLink.getAttribute('type')).toBe('video/mp4');
            expect(view.container.style.display).toBe('flex');
        });

        test('비디오 URL만 제공하면 다운로드 링크도 동일한 URL을 사용해야 한다', () => {
            const videoURL = 'http://example.com/video.mp4';
            view.showVideo(videoURL);
            expect(view.videoElement.src).toContain(videoURL);
            expect(view.downloadLink.href).toBe(videoURL);
            expect(view.downloadLink.getAttribute('download')).toBe('result.mp4');
        });

        test('빈 URL을 제공하면 비디오를 숨겨야 한다', () => {
            view.showVideo('http://example.com/video.mp4');
            view.showVideo('');
            expect(view.container.style.display).toBe('none');
            expect(view.videoElement.src).toBe('');
        });

        test('null URL을 제공하면 비디오를 숨겨야 한다', () => {
            view.showVideo('http://example.com/video.mp4');
            view.showVideo(null);
            expect(view.container.style.display).toBe('none');
        });
    });

    describe('비디오 숨기기', () => {
        test('비디오를 숨길 수 있어야 한다', () => {
            view.showVideo('http://example.com/video.mp4');
            view.hide();
            expect(view.container.style.display).toBe('none');
            expect(view.videoElement.src).toBe('');
        });
    });

    describe('리셋', () => {
        test('비디오를 리셋할 수 있어야 한다', () => {
            view.showVideo('http://example.com/video.mp4');
            view.reset();
            expect(view.container.style.display).toBe('none');
            expect(view.videoElement.src).toBe('');
        });
    });
});

