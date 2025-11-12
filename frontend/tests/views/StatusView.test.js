import { StatusView } from '../../js/views/StatusView.js';

describe('StatusView', () => {
    let view;

    beforeEach(() => {
        document.body.innerHTML = `
            <div id="status-message" hidden>
                <p data-status-message></p>
                <p data-status-hint></p>
            </div>
        `;
        view = new StatusView('status-message');
    });

    afterEach(() => {
        document.body.innerHTML = '';
    });

    test('상태 메시지를 표시할 수 있어야 한다', () => {
        view.show('분석 중', '잠시만 기다려주세요');
        const element = document.getElementById('status-message');
        expect(element.hidden).toBe(false);
        expect(element.querySelector('[data-status-message]').textContent).toBe('분석 중');
        expect(element.querySelector('[data-status-hint]').textContent).toBe('잠시만 기다려주세요');
    });

    test('상태 메시지를 숨길 수 있어야 한다', () => {
        const element = document.getElementById('status-message');
        view.show('분석 중', '잠시만 기다려주세요');
        view.hide();
        expect(element.hidden).toBe(true);
    });
});

