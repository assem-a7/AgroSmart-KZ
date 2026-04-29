// ============================================
// Charts Initialization and Configuration
// ============================================

let chartsInstances = {};

function initializeCharts() {
    initializeStatusChart();
    initializeTopSubsidiesChart();
    initializeTimeSeriesChart();
    initializePredictionChart();
    initializeAmountPredictionChart();
    initializeQualityChart();
}

// ============================================
// Chart Configuration Defaults
// ============================================

const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            display: false
        },
        tooltip: {
            backgroundColor: 'rgba(17, 24, 39, 0.95)',
            padding: 12,
            borderRadius: 8,
            titleFont: {
                size: 13,
                weight: '600'
            },
            bodyFont: {
                size: 12
            },
            displayColors: false
        }
    }
};

// ============================================
// Status Pie Chart
// ============================================

function initializeStatusChart() {
    const ctx = document.getElementById('statusChart');
    if (!ctx) return;
    
    if (chartsInstances.statusChart) {
        chartsInstances.statusChart.destroy();
    }
    
    chartsInstances.statusChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Орындалды', 'Қабылданған емес', 'Одобрена', 'Отозвано'],
            datasets: [{
                data: [79.4, 7.9, 20.8, 5.63],
                backgroundColor: [
                    '#22c55e',
                    '#ef4444',
                    '#f59e0b',
                    '#eab308'
                ],
                borderWidth: 0,
                hoverOffset: 8
            }]
        },
        options: {
            ...chartDefaults,
            cutout: '70%',
            plugins: {
                ...chartDefaults.plugins,
                tooltip: {
                    ...chartDefaults.plugins.tooltip,
                    callbacks: {
                        label: function(context) {
                            return `${context.label}: ${context.parsed}%`;
                        }
                    }
                }
            }
        }
    });
}

// ============================================
// Top Subsidies Horizontal Bar Chart
// ============================================

function initializeTopSubsidiesChart() {
    const ctx = document.getElementById('topSubsidiesChart');
    if (!ctx) return;
    
    if (chartsInstances.topSubsidiesChart) {
        chartsInstances.topSubsidiesChart.destroy();
    }
    
    const labels = [
        'Ауыл шаруашылығы техникасын сатып алу',
        'Малды асырау мен өсіру',
        'Су жүйелерін жетілдіру',
        'Тұқым материалдарын сатып алу',
        'Ауыл шаруашылығы өнімін өңдеу',
        'Органикалық шаруашылықты дамыту',
        'Жемшөп дайындау',
        'Минералды тыңайтқыштар',
        'Балық шаруашылығы',
        'Бау-бақша өнімдері'
    ];
    
    const data = [1245, 1156, 987, 854, 742, 689, 623, 578, 512, 467];
    
    chartsInstances.topSubsidiesChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: '#22c55e',
                borderRadius: 6,
                barThickness: 20
            }]
        },
        options: {
            ...chartDefaults,
            indexAxis: 'y',
            scales: {
                x: {
                    beginAtZero: true,
                    grid: {
                        display: true,
                        color: '#f3f4f6'
                    },
                    ticks: {
                        font: {
                            size: 11
                        },
                        color: '#6b7280'
                    }
                },
                y: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 11
                        },
                        color: '#374151',
                        callback: function(value) {
                            const label = this.getLabelForValue(value);
                            return label.length > 30 ? label.substring(0, 30) + '...' : label;
                        }
                    }
                }
            },
            plugins: {
                ...chartDefaults.plugins,
                tooltip: {
                    ...chartDefaults.plugins.tooltip,
                    callbacks: {
                        label: function(context) {
                            return `Өтінімдер: ${context.parsed.x}`;
                        }
                    }
                }
            }
        }
    });
}

// ============================================
// Time Series Line Chart
// ============================================

function initializeTimeSeriesChart() {
    const ctx = document.getElementById('timeSeriesChart');
    if (!ctx) return;
    
    if (chartsInstances.timeSeriesChart) {
        chartsInstances.timeSeriesChart.destroy();
    }
    
    const months = [
        'Қаң 2025', 'Ақп 2025', 'Нау 2025', 'Сәу 2025', 'Мам 2025', 'Мау 2025',
        'Шіл 2025', 'Там 2025', 'Қыр 2025', 'Қаз 2025', 'Қар 2025', 'Жел 2025'
    ];
    
    const data = [1823, 2145, 2987, 3254, 2876, 2654, 2234, 2123, 2456, 2987, 2234, 2012];
    
    chartsInstances.timeSeriesChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: months,
            datasets: [{
                label: 'Өтінімдер саны',
                data: data,
                borderColor: '#22c55e',
                backgroundColor: 'rgba(34, 197, 94, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointRadius: 4,
                pointHoverRadius: 6,
                pointBackgroundColor: '#22c55e',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }]
        },
        options: {
            ...chartDefaults,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 11
                        },
                        color: '#6b7280'
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#f3f4f6'
                    },
                    ticks: {
                        font: {
                            size: 11
                        },
                        color: '#6b7280',
                        callback: function(value) {
                            return value.toLocaleString('kk-KZ');
                        }
                    }
                }
            },
            plugins: {
                ...chartDefaults.plugins,
                legend: {
                    display: false
                },
                tooltip: {
                    ...chartDefaults.plugins.tooltip,
                    callbacks: {
                        label: function(context) {
                            return `Өтінімдер: ${context.parsed.y.toLocaleString('kk-KZ')}`;
                        }
                    }
                }
            }
        }
    });
}

// ============================================
// Prediction Chart
// ============================================

function initializePredictionChart() {
    const ctx = document.getElementById('predictionChart');
    if (!ctx) return;
    
    if (chartsInstances.predictionChart) {
        chartsInstances.predictionChart.destroy();
    }
    
    const months = ['Қаз 2025', 'Қар 2025', 'Жел 2025', 'Қаң 2026', 'Ақп 2026', 'Нау 2026', 'Сәу 2026', 'Мам 2026', 'Мау 2026'];
    const actualData = [2987, 2234, 2012, 1349, 1364, 1378, null, null, null];
    const forecastData = [null, null, null, 1349, 1364, 1378, 1392, 1405, 1420];
    
    chartsInstances.predictionChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: months,
            datasets: [
                {
                    label: 'Нақты деректер',
                    data: actualData,
                    borderColor: '#22c55e',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    pointBackgroundColor: '#22c55e',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
                },
                {
                    label: 'Болжам',
                    data: forecastData,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    borderWidth: 3,
                    borderDash: [5, 5],
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    pointBackgroundColor: '#f59e0b',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2
                }
            ]
        },
        options: {
            ...chartDefaults,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 11
                        },
                        color: '#6b7280'
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#f3f4f6'
                    },
                    ticks: {
                        font: {
                            size: 11
                        },
                        color: '#6b7280',
                        callback: function(value) {
                            return value.toLocaleString('kk-KZ');
                        }
                    }
                }
            },
            plugins: {
                ...chartDefaults.plugins,
                legend: {
                    display: false
                },
                tooltip: {
                    ...chartDefaults.plugins.tooltip,
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y ? context.parsed.y.toLocaleString('kk-KZ') : 'N/A'}`;
                        }
                    }
                }
            }
        }
    });
}

// ============================================
// Amount Prediction Chart
// ============================================

function initializeAmountPredictionChart() {
    const ctx = document.getElementById('amountPredictionChart');
    if (!ctx) return;
    
    if (chartsInstances.amountPredictionChart) {
        chartsInstances.amountPredictionChart.destroy();
    }
    
    const months = ['Қаң', 'Ақп', 'Нау', 'Сәу', 'Мам', 'Мау'];
    const subsidyData = [8.2, 9.5, 11.2, 12.8, 13.5, 14.2];
    const predictedData = [null, null, null, null, 13.5, 14.2, 15.1, 15.8];
    
    chartsInstances.amountPredictionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: months,
            datasets: [
                {
                    label: 'Нақты сома',
                    data: subsidyData,
                    backgroundColor: '#22c55e',
                    borderRadius: 6
                },
                {
                    label: 'Болжам',
                    data: predictedData,
                    backgroundColor: '#f59e0b',
                    borderRadius: 6
                }
            ]
        },
        options: {
            ...chartDefaults,
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 11
                        },
                        color: '#6b7280'
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#f3f4f6'
                    },
                    ticks: {
                        font: {
                            size: 11
                        },
                        color: '#6b7280',
                        callback: function(value) {
                            return value + ' млрд ₸';
                        }
                    }
                }
            },
            plugins: {
                ...chartDefaults.plugins,
                legend: {
                    display: false
                },
                tooltip: {
                    ...chartDefaults.plugins.tooltip,
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${context.parsed.y} млрд ₸`;
                        }
                    }
                }
            }
        }
    });
}

// ============================================
// Quality Distribution Chart
// ============================================

function initializeQualityChart() {
    const ctx = document.getElementById('qualityChart');
    if (!ctx) return;
    
    if (chartsInstances.qualityChart) {
        chartsInstances.qualityChart.destroy();
    }
    
    const ranges = ['0-20', '21-40', '41-60', '61-80', '81-100'];
    const counts = [45, 234, 956, 1284, 432];
    
    chartsInstances.qualityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ranges,
            datasets: [{
                label: 'Өтінімдер саны',
                data: counts,
                backgroundColor: [
                    '#ef4444',
                    '#f59e0b',
                    '#eab308',
                    '#22c55e',
                    '#10b981'
                ],
                borderRadius: 6
            }]
        },
        options: {
            ...chartDefaults,
            scales: {
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        font: {
                            size: 11
                        },
                        color: '#6b7280'
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#f3f4f6'
                    },
                    ticks: {
                        font: {
                            size: 11
                        },
                        color: '#6b7280',
                        callback: function(value) {
                            return value.toLocaleString('kk-KZ');
                        }
                    }
                }
            },
            plugins: {
                ...chartDefaults.plugins,
                tooltip: {
                    ...chartDefaults.plugins.tooltip,
                    callbacks: {
                        title: function(context) {
                            return `Merit Score: ${context[0].label}`;
                        },
                        label: function(context) {
                            return `Саны: ${context.parsed.y.toLocaleString('kk-KZ')}`;
                        }
                    }
                }
            }
        }
    });
}

// ============================================
// Responsive Chart Updates
// ============================================

window.addEventListener('resize', function() {
    Object.values(chartsInstances).forEach(chart => {
        if (chart) {
            chart.resize();
        }
    });
});

// ============================================
// Export
// ============================================

window.ChartsModule = {
    initializeCharts,
    chartsInstances
};

console.log('Charts module loaded');
