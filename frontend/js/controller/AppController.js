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
            '분석은 보통 1~2분 가량 소요됩니다. ',
            '절대 새로고침 하지마세요! 분석 중이던 영상 작업이 초기화됩니다.'
        );
        this.reportView.reset();
        this.videoView.reset();

        try {
            const { videoBlob, report } = await this.apiService.analyzeVideo(file);

            console.log('✅ 분석 완료:', { 
                videoBlobSize: videoBlob.size, 
                videoBlobType: videoBlob.type,
                report: report 
            });

            // 비디오 Blob 검증
            if (!videoBlob || videoBlob.size === 0) {
                throw new Error('비디오 파일이 비어있습니다.');
            }
            
            console.log('📹 Blob 정보:', {
                originalType: videoBlob.type,
                size: videoBlob.size,
                sizeMB: (videoBlob.size / 1024 / 1024).toFixed(2) + ' MB'
            });
            
            // 비디오 URL 생성 (MIME 타입 명시적으로 설정)
            const videoBlobWithType = videoBlob.type && videoBlob.type.startsWith('video/') 
                ? videoBlob 
                : new Blob([videoBlob], { type: 'video/mp4' });
            
            // Blob이 제대로 생성되었는지 확인
            if (videoBlobWithType.size === 0) {
                throw new Error('비디오 Blob 생성 실패');
            }
            
            const videoURL = URL.createObjectURL(videoBlobWithType);
            const downloadURL = URL.createObjectURL(videoBlobWithType);
            
            const baseName = (file.name || 'result').replace(/\.[^/.]+$/, '');
            
            console.log('📹 비디오 URL 생성 완료:', {
                videoURL: videoURL.substring(0, 50) + '...',
                downloadURL: downloadURL.substring(0, 50) + '...',
                blobSize: videoBlobWithType.size,
                baseName: baseName
            });
            
            this.videoModel.setVideoURL(videoURL);
            this.videoModel.setDownloadLink(downloadURL);
            this.videoModel.setDownloadFilename(`${baseName}-analysis.mp4`);

            // 리포트 설정
            this.reportModel.setReport(report);
            
            console.log('🔄 뷰 업데이트 시작');
            // 명시적으로 뷰 업데이트 (구독자 패턴이 제대로 작동하지 않을 수 있음)
            this.updateVideoView();
            this.updateReportView();
            console.log('✅ 뷰 업데이트 완료');
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

