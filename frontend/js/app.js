import { AppController } from './controllers/AppController.js';

// DOMContentLoaded 이벤트 대기
document.addEventListener('DOMContentLoaded', () => {
    try {
        const controller = new AppController();
        controller.initialize();
    } catch (error) {
        console.error('애플리케이션 초기화 실패:', error);
        alert('애플리케이션을 시작할 수 없습니다.');
    }
});