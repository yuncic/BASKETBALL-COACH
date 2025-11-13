/**
 * AppController - 애플리케이션 컨트롤러 (Model과 View를 연결)
 */
import { VideoModel } from '../models/VideoModel.js';
import { ReportModel } from '../models/ReportModel.js';
import { UploadView } from '../views/UploadView.js';
import { VideoView } from '../views/VideoView.js';
import { ReportView } from '../views/ReportView.js';
import { ApiService } from '../services/ApiService.js';
import { StatusView } from '../views/StatusView.js';

export class AppController {
    constructor() {
        // Models
        this.videoModel = new VideoModel();
        this.reportModel = new ReportModel();

        // Views
        this.uploadView = new UploadView('upload-section');
        this.videoView = new VideoView('result-section');
        this.reportView = new ReportView('report-container');
        this.statusView = new StatusView('status-message');

        // Services
        this.apiService = new ApiService();

        // State
        this.isLoading = false;
    }

    /**
     * 컨트롤러 초기화
     */
    initialize() {
        console.log('AppController 초기화 시작');
        // Views 초기화
        this.uploadView.initialize();
        console.log('UploadView 초기화 완료');
        this.videoView.initialize();
        console.log('VideoView 초기화 완료');
        this.reportView.initialize();
        console.log('ReportView 초기화 완료');

        // View 이벤트 바인딩
        this.uploadView.onFileChange((file) => {
            this.handleFileChange(file);
        });

        this.uploadView.onAnalyzeClick(() => {
            this.handleAnalyzeClick();
        });
        console.log('이벤트 바인딩 완료');

        // Model 구독
        this.videoModel.subscribe(() => {
            this.updateVideoView();
        });

        this.reportModel.subscribe(() => {
            this.updateReportView();
        });
    }

    /**
     * 파일 변경 처리
     * @param {File} file - 선택된 파일
     */
    handleFileChange(file) {
        try {
            if (file) {
                this.videoModel.setFile(file);
                this.reportModel.reset();
                this.statusView.hide();
            } else {
                this.videoModel.reset();
                this.reportModel.reset();
                this.statusView.hide();
            }
        } catch (error) {
            alert(error.message);
            this.uploadView.resetFileInput();
            this.statusView.hide();
        }
    }

    /**
     * 분석 버튼 클릭 처리
     */
    async handleAnalyzeClick() {
        console.log('분석 버튼 클릭됨');
        const file = this.videoModel.getFile();
        if (!file) {
            alert('분석할 영상을 업로드해주세요!');
            return;
        }
        console.log('파일 확인됨:', file.name);

        if (this.isLoading) {
            return;
        }

        this.isLoading = true;
        this.uploadView.setAnalyzeButtonEnabled(false);
        this.statusView.show(
            '⏳ 영상 분석 중입니다.',
            '분석은 보통 30초~1분 가량 소요됩니다.'
        );
        this.reportView.reset();
        this.videoView.reset();

        try {
            const { videoBlob, report } = await this.apiService.analyzeVideo(file);

            // 비디오 URL 생성
            const videoURL = URL.createObjectURL(videoBlob);
            const downloadURL = URL.createObjectURL(
                videoBlob.type ? videoBlob : new Blob([videoBlob], { type: 'video/mp4' })
            );
            const baseName = (file.name || 'result').replace(/\.[^/.]+$/, '');
            this.videoModel.setVideoURL(videoURL);
            this.videoModel.setDownloadLink(downloadURL);
            this.videoModel.setDownloadFilename(`${baseName}-analysis.mp4`);

            // 리포트 설정
            this.reportModel.setReport(report);
        } catch (error) {
            console.error('분석 중 오류:', error);
            alert(`분석 중 오류가 발생했습니다: ${error.message}`);
            this.reportView.showError(error.message);
        } finally {
            this.isLoading = false;
            this.uploadView.setAnalyzeButtonEnabled(true);
             this.statusView.hide();
        }
    }

    /**
     * 비디오 뷰 업데이트
     */
    updateVideoView() {
        const videoURL = this.videoModel.getVideoURL();
        const downloadLink = this.videoModel.getDownloadLink();
        const downloadName = this.videoModel.getDownloadFilename();

        if (videoURL) {
            this.videoView.showVideo(videoURL, downloadLink, downloadName);
        } else {
            this.videoView.hide();
        }
    }

    /**
     * 리포트 뷰 업데이트
     */
    updateReportView() {
        const report = this.reportModel.getReport();
        if (report) {
            this.reportView.showReport(report);
        } else {
            this.reportView.hide();
        }
    }
}

