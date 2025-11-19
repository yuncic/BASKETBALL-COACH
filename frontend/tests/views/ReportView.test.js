import { ReportView } from '../../js/views/ReportView.js';

describe('ReportView', () => {
    let container;
    let view;

    beforeEach(() => {
        // DOM 요소 생성
        document.body.innerHTML = `
            <div id="report-container" class="report-container">
                <h2>📊 분석 리포트</h2>
                <div id="report-content" class="report-content"></div>
            </div>
        `;
        container = document.getElementById('report-container');
        view = new ReportView('report-container');
        view.initialize();
    });

    afterEach(() => {
        document.body.innerHTML = '';
    });

    describe('초기화', () => {
        test('뷰가 정상적으로 초기화되어야 한다', () => {
            expect(view.reportContent).toBeTruthy();
            expect(view.container).toBe(container);
        });

        test('존재하지 않는 컨테이너 ID는 에러를 발생시켜야 한다', () => {
            expect(() => new ReportView('non-existent')).toThrow('Container with id "non-existent" not found');
        });

        test('필수 요소가 없으면 에러를 발생시켜야 한다', () => {
            document.body.innerHTML = '<div id="report-container"></div>';
            const invalidView = new ReportView('report-container');
            expect(() => invalidView.initialize()).toThrow('Report content element not found in ReportView');
        });
    });

    describe('리포트 표시', () => {
        const sampleReport = {
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
            suggestions: ['좋은 폼입니다!', '릴리즈 타이밍만 유지하면 안정적인 슛이 가능합니다.']
        };

        test('리포트를 렌더링할 수 있어야 한다', () => {
            view.showReport(sampleReport);
            const content = view.reportContent.innerHTML;
            expect(content).toContain('효율 점수: 85.5%');
            expect(content).toContain('무릎↔허리: 0.02s (양호)');
            expect(content).toContain('어깨→팔꿈치: 0.15s (적절)');
            expect(content).toContain('릴리즈 타이밍: 0.08s (적절)');
            expect(content).toContain('팔-공 정렬도: 90.5점');
            expect(content).toContain('무게중심-공 정렬도: 88.2점');
            expect(content).toContain('발사각: 45°');
            expect(content).toContain('💡 개선 제안');
            expect(content).toContain('좋은 폼입니다!');
        });

        test('개선 제안이 없으면 제안 섹션을 표시하지 않아야 한다', () => {
            const reportWithoutSuggestions = {
                ...sampleReport,
                suggestions: []
            };
            view.showReport(reportWithoutSuggestions);
            const content = view.reportContent.innerHTML;
            expect(content).not.toContain('💡 개선 제안');
        });

        test('null 리포트를 제공하면 리포트를 숨겨야 한다', () => {
            view.showReport(sampleReport);
            view.showReport(null);
            expect(view.reportContent.innerHTML).toBe('');
        });

        test('부분적인 데이터가 있어도 렌더링할 수 있어야 한다', () => {
            const partialReport = {
                eff_score: 75.0,
                metrics: {},
                alignment: {},
                suggestions: []
            };
            view.showReport(partialReport);
            const content = view.reportContent.innerHTML;
            expect(content).toContain('효율 점수: 75.0%');
        });
    });

    describe('에러 표시', () => {
        test('에러 메시지를 표시할 수 있어야 한다', () => {
            view.showError('테스트 에러');
            expect(view.reportContent.innerHTML).toContain('❌ 테스트 에러');
            expect(view.reportContent.innerHTML).toContain('error');
        });
    });

    describe('로딩 표시', () => {
        test('로딩 메시지를 표시할 수 있어야 한다', () => {
            view.showLoading('분석 중...');
            expect(view.reportContent.innerHTML).toContain('⏳ 분석 중...');
            expect(view.reportContent.innerHTML).toContain('분석은 보통 30초~1분 가량 소요됩니다.');
            expect(view.reportContent.innerHTML).toContain('loading');
        });

        test('기본 로딩 메시지를 사용할 수 있어야 한다', () => {
            view.showLoading();
            expect(view.reportContent.innerHTML).toContain('⏳ 영상 분석 중입니다.');
            expect(view.reportContent.innerHTML).toContain('분석은 보통 30초~1분 가량 소요됩니다.');
        });
    });

    describe('리포트 숨기기', () => {
        test('리포트를 숨길 수 있어야 한다', () => {
            view.showReport({
                eff_score: 85.5,
                metrics: {},
                alignment: {},
                suggestions: []
            });
            view.hide();
            expect(view.reportContent.innerHTML).toBe('');
        });
    });

    describe('리셋', () => {
        test('리포트를 리셋할 수 있어야 한다', () => {
            view.showReport({
                eff_score: 85.5,
                metrics: {},
                alignment: {},
                suggestions: []
            });
            view.reset();
            expect(view.reportContent.innerHTML).toBe('');
        });
    });
});

