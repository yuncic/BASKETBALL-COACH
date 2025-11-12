import { ReportModel } from '../../js/models/ReportModel.js';

describe('ReportModel', () => {
    let model;

    beforeEach(() => {
        model = new ReportModel();
    });

    describe('초기화', () => {
        test('모델이 정상적으로 초기화되어야 한다', () => {
            expect(model.getReport()).toBeNull();
            expect(model.getEffScore()).toBeNull();
            expect(model.getMetrics()).toBeNull();
            expect(model.getAlignment()).toBeNull();
            expect(model.getSuggestions()).toEqual([]);
        });
    });

    describe('리포트 설정', () => {
        const validReport = {
            eff_score: 85.5,
            metrics: {
                knee_hip: { gap: '0.02s', verdict: '양호' },
                shoulder_elbow: { gap: '0.15s', verdict: '적절' },
                release_timing: { gap: '0.08s', verdict: '적절' }
            },
            alignment: {
                arm_ball: 90.5,
                com_ball: 88.2,
                release_angle: 45.0
            },
            suggestions: ['좋은 폼입니다!']
        };

        test('유효한 리포트를 설정할 수 있어야 한다', () => {
            model.setReport(validReport);
            expect(model.getReport()).toEqual(validReport);
        });

        test('효율 점수를 가져올 수 있어야 한다', () => {
            model.setReport(validReport);
            expect(model.getEffScore()).toBe(85.5);
        });

        test('메트릭스를 가져올 수 있어야 한다', () => {
            model.setReport(validReport);
            const metrics = model.getMetrics();
            expect(metrics.knee_hip.gap).toBe('0.02s');
            expect(metrics.shoulder_elbow.verdict).toBe('적절');
        });

        test('정렬도 데이터를 가져올 수 있어야 한다', () => {
            model.setReport(validReport);
            const alignment = model.getAlignment();
            expect(alignment.arm_ball).toBe(90.5);
            expect(alignment.com_ball).toBe(88.2);
            expect(alignment.release_angle).toBe(45.0);
        });

        test('개선 제안을 가져올 수 있어야 한다', () => {
            model.setReport(validReport);
            expect(model.getSuggestions()).toEqual(['좋은 폼입니다!']);
        });

        test('유효하지 않은 리포트는 에러를 발생시켜야 한다', () => {
            expect(() => model.setReport(null)).toThrow('유효하지 않은 리포트 데이터입니다.');
            expect(() => model.setReport({})).toThrow('유효하지 않은 리포트 데이터입니다.');
            expect(() => model.setReport({ eff_score: 85 })).toThrow('유효하지 않은 리포트 데이터입니다.');
        });
    });

    describe('리셋', () => {
        test('리포트를 초기화할 수 있어야 한다', () => {
            const report = {
                eff_score: 85.5,
                metrics: {},
                alignment: {},
                suggestions: []
            };
            model.setReport(report);
            model.reset();
            expect(model.getReport()).toBeNull();
            expect(model.getEffScore()).toBeNull();
        });
    });

    describe('구독 패턴', () => {
        test('변경 리스너를 등록할 수 있어야 한다', () => {
            const callback = jest.fn();
            model.subscribe(callback);
            model.setReport({
                eff_score: 85.5,
                metrics: {},
                alignment: {},
                suggestions: []
            });
            expect(callback).toHaveBeenCalled();
        });

        test('변경 리스너를 제거할 수 있어야 한다', () => {
            const callback = jest.fn();
            model.subscribe(callback);
            model.unsubscribe(callback);
            model.setReport({
                eff_score: 85.5,
                metrics: {},
                alignment: {},
                suggestions: []
            });
            expect(callback).not.toHaveBeenCalled();
        });
    });
});

