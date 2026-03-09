/**
 * StatusView - 분석 상태 메시지 관리
 */
export class StatusView {
    constructor(elementId) {
        this.element = document.getElementById(elementId);
        if (!this.element) {
            throw new Error(`Status element with id "${elementId}" not found`);
        }
        this.messageEl = this.element.querySelector('[data-status-message]');
        this.hintEl = this.element.querySelector('[data-status-hint]');
    }

    /**
     * 상태 메시지 표시
     * @param {string} message - 주요 메시지
     * @param {string} hint - 부가 설명
     */
    show(message, hint) {
        if (this.messageEl) {
            this.messageEl.textContent = message;
        }
        if (this.hintEl) {
            this.hintEl.textContent = hint;
        }
        this.element.hidden = false;
    }

    /**
     * 상태 메시지 숨김
     */
    hide() {
        this.element.hidden = true;
    }
}

