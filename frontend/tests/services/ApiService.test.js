import { ApiService } from '../../js/services/ApiService.js';

describe('ApiService', () => {
    let service;
    let fetchMock;

    beforeEach(() => {
        service = new ApiService();
        global.fetch = jest.fn();
        fetchMock = global.fetch;
    });

    afterEach(() => {
        jest.restoreAllMocks();
    });

    describe('초기화', () => {
        test('서비스가 정상적으로 초기화되어야 한다', () => {
            expect(service.baseURL).toBe('');
        });

        test('baseURL을 설정할 수 있어야 한다', () => {
            const serviceWithURL = new ApiService('http://localhost:8000');
            expect(serviceWithURL.baseURL).toBe('http://localhost:8000');
        });
    });

    describe('비디오 분석', () => {
        test('파일이 없으면 에러를 발생시켜야 한다', async () => {
            await expect(service.analyzeVideo(null)).rejects.toThrow('분석할 영상을 업로드해주세요!');
        });

        test('비디오 분석 API를 호출할 수 있어야 한다', async () => {
            const file = new File(['test'], 'test.mp4', { type: 'video/mp4' });
            const videoBlob = new Blob(['video'], { type: 'video/mp4' });
            const report = {
                eff_score: 85.5,
                metrics: {},
                alignment: {},
                suggestions: []
            };
            const base64Report = btoa(JSON.stringify(report));

            fetchMock.mockResolvedValueOnce({
                ok: true,
                headers: {
                    get: jest.fn((header) => {
                        if (header === 'X-Report-Base64' || header === 'x-report-base64') {
                            return base64Report;
                        }
                        return null;
                    })
                },
                blob: jest.fn().mockResolvedValueOnce(videoBlob)
            });

            const result = await service.analyzeVideo(file);
            expect(result.videoBlob).toBe(videoBlob);
            expect(result.report).toEqual(report);
            expect(fetchMock).toHaveBeenCalledWith(
                '/api/analyze',
                expect.objectContaining({
                    method: 'POST',
                    body: expect.any(FormData)
                })
            );
        });

        test('API 호출 실패 시 에러를 발생시켜야 한다', async () => {
            const file = new File(['test'], 'test.mp4', { type: 'video/mp4' });
            fetchMock.mockResolvedValueOnce({
                ok: false,
                status: 500
            });

            await expect(service.analyzeVideo(file)).rejects.toThrow('분석 실패: 500');
        });
    });

    describe('리포트 가져오기', () => {
        test('경로로 리포트를 가져올 수 있어야 한다', async () => {
            const reportPath = '/path/to/report.json';
            const report = {
                eff_score: 85.5,
                metrics: {},
                alignment: {},
                suggestions: []
            };

            fetchMock.mockResolvedValueOnce({
                ok: true,
                text: jest.fn().mockResolvedValueOnce(JSON.stringify(report))
            });

            const result = await service.getReportByPath(reportPath);
            expect(result).toEqual(report);
            expect(fetchMock).toHaveBeenCalledWith(
                `/api/report?path=${encodeURIComponent(reportPath)}`
            );
        });

        test('리포트 가져오기 실패 시 에러를 발생시켜야 한다', async () => {
            const reportPath = '/path/to/report.json';
            fetchMock.mockResolvedValueOnce({
                ok: false,
                status: 404
            });

            await expect(service.getReportByPath(reportPath)).rejects.toThrow('리포트 가져오기 실패: 404');
        });
    });

    describe('Base64 디코딩', () => {
        test('Base64 문자열을 리포트로 디코딩할 수 있어야 한다', () => {
            const report = {
                eff_score: 85.5,
                metrics: {},
                alignment: {},
                suggestions: []
            };
            const base64String = btoa(JSON.stringify(report));
            const result = service.decodeReportFromBase64(base64String);
            expect(result).toEqual(report);
        });

        test('유효하지 않은 Base64 문자열은 에러를 발생시켜야 한다', () => {
            expect(() => service.decodeReportFromBase64('invalid-base64!!!')).toThrow();
        });
    });
});

