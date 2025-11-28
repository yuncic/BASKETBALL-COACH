import { VideoModel } from '../../js/models/VideoModel.js';

describe('VideoModel', () => {
    let model;

    beforeEach(() => {
        model = new VideoModel();
    });

    describe('초기화', () => {
        test('모델이 정상적으로 초기화되어야 한다', () => {
            expect(model.getFile()).toBeNull();
            expect(model.getVideoURL()).toBeNull();
            expect(model.getDownloadLink()).toBeNull();
            expect(model.getDownloadFilename()).toBe('result.mp4');
        });
    });

    describe('파일 설정', () => {
        test('유효한 비디오 파일을 설정할 수 있어야 한다', () => {
            const file = new File(['test'], 'test.mp4', { type: 'video/mp4' });
            model.setFile(file);
            expect(model.getFile()).toBe(file);
        });

        test('유효하지 않은 파일 형식은 에러를 발생시켜야 한다', () => {
            const file = new File(['test'], 'test.txt', { type: 'text/plain' });
            expect(() => model.setFile(file)).toThrow('지원하지 않는 비디오 파일 형식입니다.');
        });

        test('파일 확장자로도 유효성을 검사할 수 있어야 한다', () => {
            const file = new File(['test'], 'test.mov', { type: '' });
            model.setFile(file);
            expect(model.getFile()).toBe(file);
        });

        test('파일을 설정하면 비디오 URL과 다운로드 정보가 초기화되어야 한다', () => {
            model.setVideoURL('http://example.com/video.mp4');
            model.setDownloadLink('http://example.com/video.mp4');
            model.setDownloadFilename('custom.mp4');
            const file = new File(['test'], 'test.mp4', { type: 'video/mp4' });
            model.setFile(file);
            expect(model.getVideoURL()).toBeNull();
            expect(model.getDownloadLink()).toBeNull();
            expect(model.getDownloadFilename()).toBe('result.mp4');
        });
    });

    describe('비디오 URL 설정', () => {
        test('비디오 URL을 설정할 수 있어야 한다', () => {
            const url = 'http://example.com/video.mp4';
            model.setVideoURL(url);
            expect(model.getVideoURL()).toBe(url);
        });

        test('다운로드 정보를 설정할 수 있어야 한다', () => {
            const url = 'http://example.com/video.mp4';
            model.setDownloadLink(url);
            model.setDownloadFilename('test.mp4');
            expect(model.getDownloadLink()).toBe(url);
            expect(model.getDownloadFilename()).toBe('test.mp4');
        });
    });

    describe('리셋', () => {
        test('모든 상태를 초기화할 수 있어야 한다', () => {
            const file = new File(['test'], 'test.mp4', { type: 'video/mp4' });
            model.setFile(file);
            model.setVideoURL('http://example.com/video.mp4');
            model.setDownloadLink('http://example.com/video.mp4');
            model.setDownloadFilename('test.mp4');
            model.reset();
            expect(model.getFile()).toBeNull();
            expect(model.getVideoURL()).toBeNull();
            expect(model.getDownloadLink()).toBeNull();
            expect(model.getDownloadFilename()).toBe('result.mp4');
        });
    });

    describe('구독 패턴', () => {
        test('변경 리스너를 등록할 수 있어야 한다', () => {
            const callback = jest.fn();
            model.subscribe(callback);
            model.setFile(new File(['test'], 'test.mp4', { type: 'video/mp4' }));
            expect(callback).toHaveBeenCalled();
        });

        test('변경 리스너를 제거할 수 있어야 한다', () => {
            const callback = jest.fn();
            model.subscribe(callback);
            model.unsubscribe(callback);
            model.setFile(new File(['test'], 'test.mp4', { type: 'video/mp4' }));
            expect(callback).not.toHaveBeenCalled();
        });

        test('에러가 발생해도 다른 리스너는 계속 호출되어야 한다', () => {
            const callback1 = jest.fn(() => { throw new Error('Test error'); });
            const callback2 = jest.fn();
            model.subscribe(callback1);
            model.subscribe(callback2);
            model.setFile(new File(['test'], 'test.mp4', { type: 'video/mp4' }));
            expect(callback1).toHaveBeenCalled();
            expect(callback2).toHaveBeenCalled();
        });
    });
});

