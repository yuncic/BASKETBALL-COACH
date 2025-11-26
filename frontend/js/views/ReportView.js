
export class ReportView {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container with id "${containerId}" not found`);
        }
        this.reportContent = null;
    }

    initialize() {
        this.reportContent = this.container.querySelector('#report-content');
        if (!this.reportContent) {
            throw new Error('Report content element not found in ReportView');
        }
    }

    /**
     * 리포트 표시
     * @param {Object} report - 리포트 데이터
     */
    showReport(report) {
        if (!report) {
            this.hide();
            return;
        }

        const html = this.renderReport(report);
        this.reportContent.innerHTML = html;
    }

    /**
     * 리포트 렌더링
     * @param {Object} report - 리포트 데이터
     * @returns {string} HTML 문자열
     */
    
    renderReport(report) {
        const effScore = report.eff_score?.toFixed?.(1) ?? 'N/A';
        const metrics = report.metrics || {};
        const suggestions = report.suggestions || [];
        const powerTransfer = report.power_transfer ?? 0;
        const releaseAngle = report.release_angle ?? 0;
        
        let html = `
            <p>효율 점수: ${effScore}%</p>
            <p>무릎↔허리: ${metrics.knee_hip?.gap ?? '-'} (${metrics.knee_hip?.verdict ?? '-'})</p>
            <p>어깨→팔꿈치: ${metrics.shoulder_elbow?.gap ?? '-'} (${metrics.shoulder_elbow?.verdict ?? '-'})</p>
            <p>릴리즈 타이밍: ${metrics.release_timing?.gap ?? '-'} (${metrics.release_timing?.verdict ?? '-'})</p>
            <p>힘 전달 효율: ${powerTransfer}%</p>
            <p>발사각: ${releaseAngle}°</p>
        `;

        if (suggestions.length > 0) {
            html += '<h3>💡 개선 제안</h3>';
            html += '<ul>';
            suggestions.forEach(suggestion => {
                html += `<li>${suggestion}</li>`;
            });
            html += '</ul>';
        }

        return html;
    }

    /**
     * 리포트 숨기기
     */
    hide() {
        if (this.reportContent) {
            this.reportContent.innerHTML = '';
        }
    }

    /**
     * 리포트 초기화
     */
    reset() {
        this.hide();
    }

    /**
     * 에러 메시지 표시
     * @param {string} message - 에러 메시지
     */
    showError(message) {
        if (this.reportContent) {
            this.reportContent.innerHTML = `<p class="error">❌ ${message}</p>`;
        }
    }

    /**
     * 로딩 메시지 표시
     * @param {string} message - 로딩 메시지
     */
    showLoading(message = '영상 분석 중입니다.') {
        if (this.reportContent) {
            this.reportContent.innerHTML = `
                <div class="loading-card">
                    <p class="loading">⏳ ${message}</p>
                    <p class="loading-hint">분석은 보통 30초~1분 가량 소요됩니다.</p>
                </div>
            `;
        }
    }
}

