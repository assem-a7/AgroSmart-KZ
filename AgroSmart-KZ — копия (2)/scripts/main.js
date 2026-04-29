// ============================================
// AgroSmart KZ - Main JavaScript
// ============================================

// Global State
const AppState = {
    currentSection: 'dashboard-overview',
    chatOpen: false,
    sidebarOpen: false
};

// ============================================
// DOM Ready
// ============================================

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    initializeNavigation();
    initializeForms();
    initializeChat();
    initializeSliders();
    initializeQuickActions();
    
    // Load initial charts
    if (typeof initializeCharts === 'function') {
        initializeCharts();
    }
}

// ============================================
// Navigation System
// ============================================

function initializeNavigation() {
    // Sidebar navigation
    const sidebarLinks = document.querySelectorAll('.sidebar-link');
    sidebarLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);
            navigateToSection(targetId);
            
            // Update active state
            sidebarLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });
    
    // Main nav links
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            navLinks.forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });
}

function navigateToSection(sectionId) {
    // Hide all sections
    const sections = document.querySelectorAll('.content-section');
    sections.forEach(section => {
        section.classList.remove('active');
    });
    
    // Show target section
    const targetSection = document.getElementById(sectionId + '-section') || 
                         document.getElementById(sectionId);
    if (targetSection) {
        targetSection.classList.add('active');
        AppState.currentSection = sectionId;
        
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

// ============================================
// Form Handling
// ============================================

function initializeForms() {
    // Verification form
    const verificationForm = document.getElementById('verificationForm');
    if (verificationForm) {
        verificationForm.addEventListener('submit', handleVerificationSubmit);
    }
    
    // Soil analysis form
    const soilImageUpload = document.getElementById('soilImageUpload');
    if (soilImageUpload) {
        soilImageUpload.addEventListener('change', handleSoilImageUpload);
    }
    
    const analyzeSoilBtn = document.getElementById('analyzeSoilBtn');
    if (analyzeSoilBtn) {
        analyzeSoilBtn.addEventListener('click', handleSoilAnalysis);
    }
}

function handleVerificationSubmit(e) {
    e.preventDefault();
    
    // Get form data
    const formData = {
        region: document.getElementById('region').value,
        cropType: document.getElementById('crop-type').value,
        area: parseFloat(document.getElementById('area').value),
        subsidyAmount: parseFloat(document.getElementById('subsidy-amount').value),
        applicationDate: document.getElementById('application-date').value,
        farmerId: document.getElementById('farmer-id').value,
        landQuality: parseFloat(document.getElementById('land-quality').value)
    };
    
    // Show loading state
    showLoadingState('Верификация өтілуде...');
    
    // Simulate API call
    setTimeout(() => {
        const result = simulateVerification(formData);
        displayVerificationResult(result);
        hideLoadingState();
    }, 2000);
}

function simulateVerification(data) {
    // Simple scoring logic
    let score = 70;
    
    if (data.landQuality > 70) score += 10;
    if (data.area > 100) score += 5;
    if (data.subsidyAmount < 10000000) score += 10;
    
    const riskLevel = score > 85 ? 'low' : score > 70 ? 'medium' : 'high';
    
    return {
        score: Math.min(score, 100),
        riskLevel: riskLevel,
        approved: score > 70,
        recommendations: generateRecommendations(riskLevel, data)
    };
}

function generateRecommendations(riskLevel, data) {
    const recommendations = [];
    
    if (data.landQuality < 60) {
        recommendations.push('Жер сапасын жақсарту үшін тыңайтқыш қолдану ұсынылады');
    }
    
    if (data.area > 500) {
        recommendations.push('Үлкен аудан үшін қосымша мониторинг қажет');
    }
    
    if (riskLevel === 'high') {
        recommendations.push('Қосымша құжаттама талап етіледі');
    }
    
    return recommendations;
}

function displayVerificationResult(result) {
    const resultContainer = document.getElementById('verificationResult');
    
    const statusClass = result.approved ? 'success' : 'warning';
    const statusText = result.approved ? 'Мақұлданды' : 'Қосымша тексеру қажет';
    const statusIcon = result.approved ? 'fa-check-circle' : 'fa-exclamation-triangle';
    
    const html = `
        <div class="verification-summary ${statusClass}">
            <div class="verification-header">
                <i class="fas ${statusIcon}"></i>
                <h3>${statusText}</h3>
            </div>
            <div class="verification-details">
                <div class="detail-item">
                    <span class="detail-label">Скор:</span>
                    <span class="detail-value">${result.score}/100</span>
                </div>
                <div class="detail-item">
                    <span class="detail-label">Тәуекел деңгейі:</span>
                    <span class="detail-value risk-${result.riskLevel}">${getRiskText(result.riskLevel)}</span>
                </div>
            </div>
            ${result.recommendations.length > 0 ? `
                <div class="recommendations">
                    <h4>Ұсыныстар:</h4>
                    <ul>
                        ${result.recommendations.map(rec => `<li>${rec}</li>`).join('')}
                    </ul>
                </div>
            ` : ''}
        </div>
    `;
    
    resultContainer.innerHTML = html;
    resultContainer.classList.remove('hidden');
}

function getRiskText(level) {
    const texts = {
        'low': 'Төмен',
        'medium': 'Орташа',
        'high': 'Жоғары'
    };
    return texts[level] || level;
}

// ============================================
// Soil Analysis
// ============================================

function handleSoilImageUpload(e) {
    const files = e.target.files;
    if (files.length > 0) {
        const uploadArea = document.getElementById('uploadArea');
        uploadArea.innerHTML = `
            <div class="upload-icon">
                <i class="fas fa-check-circle" style="color: var(--success);"></i>
            </div>
            <p class="upload-text">${files.length} файл таңдалды</p>
            <button class="btn-secondary" onclick="document.getElementById('soilImageUpload').click()">
                <i class="fas fa-sync-alt"></i>
                Басқа файл таңдау
            </button>
        `;
    }
}

function handleSoilAnalysis() {
    const fileInput = document.getElementById('soilImageUpload');
    const region = document.getElementById('land-region').value;
    
    if (fileInput.files.length === 0) {
        showNotification('Алдымен фото жүктеңіз', 'warning');
        return;
    }
    
    if (!region) {
        showNotification('Облысты таңдаңыз', 'warning');
        return;
    }
    
    showLoadingState('Топырақ талдануда...');
    
    setTimeout(() => {
        const result = simulateSoilAnalysis();
        displaySoilAnalysisResult(result);
        hideLoadingState();
    }, 2500);
}

function simulateSoilAnalysis() {
    return {
        soilType: 'Қара топырақ (Чернозём)',
        fertility: 85,
        ph: 6.8,
        nitrogen: 'Орташа',
        phosphorus: 'Жоғары',
        potassium: 'Орташа',
        recommendations: [
            'Топырақ құрамы бидай өсіру үшін өте қолайлы',
            'Азот тыңайтқышын қосу арқылы өнімділікті арттыруға болады',
            'Құрғақшылыққа төзімділікті арттыру үшін органикалық тыңайтқыш ұсынылады'
        ]
    };
}

function displaySoilAnalysisResult(result) {
    showNotification(`Талдау аяқталды! Топырақ түрі: ${result.soilType}`, 'success');
}

// ============================================
// Input Controls
// ============================================

window.adjustValue = function(inputId, delta) {
    const input = document.getElementById(inputId);
    if (input) {
        const currentValue = parseFloat(input.value) || 0;
        const step = parseFloat(input.step) || 1;
        const newValue = currentValue + delta;
        const min = parseFloat(input.min);
        const max = parseFloat(input.max);
        
        if ((isNaN(min) || newValue >= min) && (isNaN(max) || newValue <= max)) {
            input.value = newValue.toFixed(2);
        }
    }
};

// ============================================
// Sliders
// ============================================

function initializeSliders() {
    const predMonthsSlider = document.getElementById('pred-months');
    if (predMonthsSlider) {
        predMonthsSlider.addEventListener('input', function() {
            const valueSpan = this.parentElement.querySelector('.slider-value');
            if (valueSpan) {
                valueSpan.textContent = `${this.value} ай`;
            }
        });
    }
}

// ============================================
// Quick Actions
// ============================================

function initializeQuickActions() {
    const quickActionBtns = document.querySelectorAll('.quick-action-btn');
    quickActionBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const question = this.textContent.trim();
            sendChatMessage(question, true);
        });
    });
}

// ============================================
// Notifications
// ============================================

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${getNotificationIcon(type)}"></i>
        <span>${message}</span>
    `;
    
    notification.style.cssText = `
        position: fixed;
        top: 90px;
        right: 24px;
        background: white;
        padding: 16px 24px;
        border-radius: 8px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
        gap: 12px;
        z-index: 10000;
        animation: slideInRight 0.3s ease-out;
        border-left: 4px solid ${getNotificationColor(type)};
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

function getNotificationIcon(type) {
    const icons = {
        'success': 'check-circle',
        'warning': 'exclamation-triangle',
        'error': 'times-circle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

function getNotificationColor(type) {
    const colors = {
        'success': '#22c55e',
        'warning': '#f59e0b',
        'error': '#ef4444',
        'info': '#3b82f6'
    };
    return colors[type] || '#3b82f6';
}

// ============================================
// Loading State
// ============================================

function showLoadingState(message = 'Жүктелуде...') {
    const loader = document.createElement('div');
    loader.id = 'globalLoader';
    loader.innerHTML = `
        <div style="position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.5); display: flex; align-items: center; justify-content: center; z-index: 10000;">
            <div style="background: white; padding: 32px; border-radius: 12px; text-align: center; min-width: 200px;">
                <div style="width: 48px; height: 48px; border: 4px solid #e5e7eb; border-top-color: #22c55e; border-radius: 50%; margin: 0 auto 16px; animation: spin 1s linear infinite;"></div>
                <p style="margin: 0; color: #374151; font-weight: 500;">${message}</p>
            </div>
        </div>
    `;
    document.body.appendChild(loader);
}

function hideLoadingState() {
    const loader = document.getElementById('globalLoader');
    if (loader) {
        loader.remove();
    }
}

// Add spin animation
const style = document.createElement('style');
style.textContent = `
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .verification-summary {
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: white;
    }
    
    .verification-summary.success {
        background: linear-gradient(135deg, #ecfdf5 0%, #ffffff 100%);
        border-color: #22c55e;
    }
    
    .verification-summary.warning {
        background: linear-gradient(135deg, #fffbeb 0%, #ffffff 100%);
        border-color: #f59e0b;
    }
    
    .verification-header {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 24px;
    }
    
    .verification-header i {
        font-size: 32px;
        color: #22c55e;
    }
    
    .verification-summary.warning .verification-header i {
        color: #f59e0b;
    }
    
    .verification-header h3 {
        margin: 0;
        font-size: 20px;
        color: #111827;
    }
    
    .verification-details {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    
    .detail-item {
        display: flex;
        justify-content: space-between;
        padding: 12px;
        background: white;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    
    .detail-label {
        font-size: 14px;
        color: #6b7280;
    }
    
    .detail-value {
        font-size: 16px;
        font-weight: 600;
        color: #111827;
    }
    
    .detail-value.risk-low {
        color: #22c55e;
    }
    
    .detail-value.risk-medium {
        color: #f59e0b;
    }
    
    .detail-value.risk-high {
        color: #ef4444;
    }
    
    .recommendations {
        padding: 16px;
        background: white;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    
    .recommendations h4 {
        font-size: 14px;
        font-weight: 600;
        color: #111827;
        margin-bottom: 12px;
    }
    
    .recommendations ul {
        margin: 0;
        padding-left: 20px;
    }
    
    .recommendations li {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 8px;
        line-height: 1.5;
    }
    
    .recommendations li:last-child {
        margin-bottom: 0;
    }
`;
document.head.appendChild(style);

// ============================================
// Utility Functions
// ============================================

function formatNumber(num) {
    return new Intl.NumberFormat('kk-KZ').format(num);
}

function formatCurrency(num) {
    return new Intl.NumberFormat('kk-KZ', {
        style: 'currency',
        currency: 'KZT',
        minimumFractionDigits: 0
    }).format(num);
}

function formatDate(date) {
    return new Intl.DateTimeFormat('kk-KZ', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    }).format(new Date(date));
}

// ============================================
// Export for use in other files
// ============================================

window.AgroSmartApp = {
    navigateToSection,
    showNotification,
    showLoadingState,
    hideLoadingState,
    formatNumber,
    formatCurrency,
    formatDate
};

console.log('AgroSmart KZ initialized successfully');
