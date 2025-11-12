
export class ApiService {
    constructor(baseURL = '') {
        this.baseURL = baseURL;
    }

    /**
     * 비디오 분석 요청
     * @param {File} file - 업로드할 비디오 파일
     * @returns {Promise<{videoBlob: Blob, report: Object}>} 분석 결과 비디오와 리포트
     */
    async analyzeVideo(file) {
        if (!file) {
            throw new Error('분석할 영상을 업로드해주세요!');
        }

        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${this.baseURL}/api/analyze`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`분석 실패: ${response.status}`);
        }

        // 비디오 Blob 받기
        const videoBlob = await response.blob();

        // 리포트 가져오기 (헤더에서)
        const pathHeader = response.headers.get('X-Report-Path') || response.headers.get('x-report-path');
        const b64Header = response.headers.get('X-Report-Base64') || response.headers.get('x-report-base64');

        let report = null;

        // 방법 1: 경로로 리포트 가져오기
        if (pathHeader) {
            try {
                report = await this.getReportByPath(pathHeader);
            } catch (error) {
                console.warn('Failed to fetch report by path:', error);
            }
        }

        // 방법 2: Base64로 리포트 복원
        if (!report && b64Header) {
            try {
                report = this.decodeReportFromBase64(b64Header);
            } catch (error) {
                console.warn('Failed to decode report from base64:', error);
            }
        }

        if (!report) {
            throw new Error('리포트를 가져올 수 없습니다.');
        }

        return {
            videoBlob,
            report,
        };
    }

    /**
     * 경로로 리포트 가져오기
     * @param {string} path - 리포트 파일 경로
     * @returns {Promise<Object>} 리포트 데이터
     */
    async getReportByPath(path) {
        const encodedPath = encodeURIComponent(path);
        const response = await fetch(`${this.baseURL}/api/report?path=${encodedPath}`);

        if (!response.ok) {
            throw new Error(`리포트 가져오기 실패: ${response.status}`);
        }

        const text = await response.text();
        const data = JSON.parse(text);
        return data;
    }

    /**
     * Base64 문자열에서 리포트 디코딩
     * @param {string} base64String - Base64로 인코딩된 리포트
     * @returns {Object} 리포트 데이터
     */
    decodeReportFromBase64(base64String) {
        try {
            // Base64 → Uint8Array → UTF-8 복원
            const binary = atob(base64String);
            const bytes = Uint8Array.from(binary, c => c.charCodeAt(0));
            const text = new TextDecoder('utf-8').decode(bytes);
            const parsed = JSON.parse(text);
            return parsed;
        } catch (error) {
            throw new Error(`Base64 디코딩 실패: ${error.message}`);
        }
    }
}

