/**
 * ReportModel - 리포트 데이터 관리
 */
export class ReportModel {
    constructor() {
        this.report = null;
        this.listeners = [];
    }

    /**
     * 리포트 설정
     * @param {Object} report - 리포트 데이터
     */
    setReport(report) {
        if (!this.isValidReport(report)) {
            throw new Error("유효하지 않은 리포트 데이터입니다.");
        }
        this.report = report;
        this.notifyListeners();
    }

    /**
     * 리포트 초기화
     */
    reset() {
        this.report = null;
        this.notifyListeners();
    }

    /**
     * 리포트 데이터 유효성 검사
     * @param {Object} report - 검사할 리포트 데이터
     * @returns {boolean} 유효한 리포트인지 여부
     */
    isValidReport(report) {
        if (!report || typeof report !== 'object') {
            return false;
        }
        // 기본 구조 검사
        return (
            typeof report.eff_score === 'number' &&
            report.metrics &&
            Array.isArray(report.suggestions)
        );
    }

    /**
     * 리포트 가져오기
     * @returns {Object|null} 현재 리포트 데이터
     */
    getReport() {
        return this.report;
    }

    /**
     * 효율 점수 가져오기
     * @returns {number|null} 효율 점수
     */
    getEffScore() {
        return this.report?.eff_score ?? null;
    }

    /**
     * 메트릭스 가져오기
     * @returns {Object|null} 메트릭스 데이터
     */
    getMetrics() {
        return this.report?.metrics ?? null;
    }

    /**
     * 힘 전달 효율 가져오기
     * @returns {number|null} 힘 전달 효율
     */
    getPowerTransfer() {
        return this.report?.power_transfer ?? null;
    }

    /**
     * 개선 제안 가져오기
     * @returns {Array} 개선 제안 목록
     */
    getSuggestions() {
        return this.report?.suggestions ?? [];
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
                console.error('ReportModel listener error:', error);
            }
        });
    }
}

