// ============================================
// AI Chat Functionality
// ============================================

let chatHistory = [];
let isChatOpen = false;

function initializeChat() {
    const chatToggleBtn = document.getElementById('chatToggleBtn');
    const closeChatBtn = document.getElementById('closeChatBtn');
    const chatPanel = document.getElementById('aiChatPanel');
    const sendMessageBtn = document.getElementById('sendMessageBtn');
    const chatInput = document.getElementById('chatInput');
    
    // Toggle chat panel
    if (chatToggleBtn) {
        chatToggleBtn.addEventListener('click', function() {
            toggleChat();
        });
    }
    
    // Close chat
    if (closeChatBtn) {
        closeChatBtn.addEventListener('click', function() {
            closeChat();
        });
    }
    
    // Send message on button click
    if (sendMessageBtn) {
        sendMessageBtn.addEventListener('click', function() {
            sendMessage();
        });
    }
    
    // Send message on Enter key
    if (chatInput) {
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
}

function toggleChat() {
    const chatPanel = document.getElementById('aiChatPanel');
    const chatToggleBtn = document.getElementById('chatToggleBtn');
    
    isChatOpen = !isChatOpen;
    
    if (isChatOpen) {
        chatPanel.classList.add('open');
        chatToggleBtn.classList.add('active');
        
        // Remove notification badge
        const badge = chatToggleBtn.querySelector('.chat-badge');
        if (badge) {
            badge.style.display = 'none';
        }
    } else {
        chatPanel.classList.remove('open');
        chatToggleBtn.classList.remove('active');
    }
}

function closeChat() {
    const chatPanel = document.getElementById('aiChatPanel');
    const chatToggleBtn = document.getElementById('chatToggleBtn');
    
    isChatOpen = false;
    chatPanel.classList.remove('open');
    chatToggleBtn.classList.remove('active');
}

function sendMessage() {
    const chatInput = document.getElementById('chatInput');
    const message = chatInput.value.trim();
    
    if (!message) return;
    
    // Add user message
    addMessage(message, 'user');
    
    // Clear input
    chatInput.value = '';
    
    // Show typing indicator
    showTypingIndicator();
    
    // Simulate AI response
    setTimeout(() => {
        const response = generateAIResponse(message);
        hideTypingIndicator();
        addMessage(response, 'ai');
    }, 1500);
}

function sendChatMessage(message, isQuickAction = false) {
    if (!isChatOpen) {
        toggleChat();
    }
    
    // Add user message
    addMessage(message, 'user');
    
    // Show typing indicator
    showTypingIndicator();
    
    // Simulate AI response
    setTimeout(() => {
        const response = generateAIResponse(message);
        hideTypingIndicator();
        addMessage(response, 'ai');
    }, 1500);
}

function addMessage(text, sender) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const now = new Date();
    const timeString = now.getHours().toString().padStart(2, '0') + ':' + 
                      now.getMinutes().toString().padStart(2, '0');
    
    messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-${sender === 'ai' ? 'robot' : 'user'}"></i>
        </div>
        <div class="message-content">
            <p>${text}</p>
            <div class="message-time">${timeString}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    
    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Store in history
    chatHistory.push({
        sender: sender,
        text: text,
        timestamp: now
    });
}

function showTypingIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typingIndicator';
    typingDiv.className = 'message ai-message';
    
    typingDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

function generateAIResponse(userMessage) {
    const lowerMessage = userMessage.toLowerCase();
    
    // Subsidy-related questions
    if (lowerMessage.includes('субсидия') && lowerMessage.includes('шарт')) {
        return 'Субсидия алу үшін негізгі шарттар:\n\n1. Тіркелген шаруашылық болуы керек\n2. Жер учаскесінің заңды құжаттары\n3. Бизнес-жоспар ұсыну\n4. Алдыңғы жылдардағы қызмет нәтижелері\n5. Қажетті лицензиялар мен рұқсаттар\n\nҚосымша ақпарат үшін жергілікті ауыл шаруашылығы бөліміне хабарласыңыз.';
    }
    
    // Prediction questions
    if (lowerMessage.includes('болжам') || lowerMessage.includes('2025')) {
        return '2025 жылға арналған болжам бойынша:\n\n📈 Өтінімдер саны: 15-20% өсу күтілуде\n💰 Жалпы бюджет: 150-160 млрд ₸ аралығында\n🌾 Басымдық бағыттар: бидай, мал шаруашылығы, су жүйелері\n\nАI модельдеріміз Exponential Smoothing алгоритмін қолданады және 85% дәлдікпен болжам береді.';
    }
    
    // Region questions
    if (lowerMessage.includes('облыс') || lowerMessage.includes('көп')) {
        return 'Ең көп субсидия алатын облыстар (2025):\n\n1. 🥇 Түркістан облысы - 11.5 млрд ₸\n2. 🥈 Солтүстік-Қазақстан - 10.5 млрд ₸\n3. 🥉 Қостанай облысы - 7.1 млрд ₸\n\nБұл облыстарда ауыл шаруашылығы дамыған және Merit Score жоғары.';
    }
    
    // Soil quality questions
    if (lowerMessage.includes('топырақ') || lowerMessage.includes('жақсарт')) {
        return 'Топырақ сапасын жақсарту жолдары:\n\n🌱 Органикалық тыңайтқыштар қолдану\n💧 Дұрыс суару жүйесін қолдану\n🔄 Егіс айналымын сақтау\n🧪 Топырақ талдауын жүргізу\n🌾 Жабынды дақылдар егу\n\nБізде AI топырақ талдау құралы бар - "Жер талдауы" бөліміне өтіңіз!';
    }
    
    // Merit score questions
    if (lowerMessage.includes('merit') || lowerMessage.includes('скор')) {
        return 'Merit Score - бұл субсидия өтінімінің сапасын бағалайтын көрсеткіш (0-100):\n\n✅ 80-100: Өте жоғары сапа\n🟢 60-79: Жақсы деңгей\n🟡 40-59: Орташа деңгей\n🔴 0-39: Төмен деңгей\n\nСкорды жақсарту үшін: жер сапасын арттыру, дұрыс құжаттама ұсыну, және алдыңғы жылдардағы нәтижелерді көрсету керек.';
    }
    
    // Risk questions
    if (lowerMessage.includes('тәуекел') || lowerMessage.includes('қауіп')) {
        return 'Тәуекел деңгейлері және ұсыныстар:\n\n🟢 Төмен тәуекел (<20%): Қалыпты өтінім процесі\n🟡 Орташа тәуекел (20-50%): Қосымша құжаттама қажет\n🔴 Жоғары тәуекел (>50%): Толық тексеру және мониторинг\n\nАI моделіміз Isolation Forest + XGBoost алгоритмдерін қолданып, аномалияларды анықтайды.';
    }
    
    // Verification questions
    if (lowerMessage.includes('верификация') || lowerMessage.includes('тексер')) {
        return 'Субсидия верификациясы автоматты түрде жүргізіледі:\n\n1️⃣ Өтінім деректерін енгізіңіз\n2️⃣ AI модель талдау жүргізеді\n3️⃣ Merit Score есептеледі\n4️⃣ Тәуекел деңгейі анықталады\n5️⃣ Ұсыныстар беріледі\n\n"Субсидия верификациясы" бөліміне өтіп, қазір тексере аласыз!';
    }
    
    // Weather/climate questions
    if (lowerMessage.includes('ауа райы') || lowerMessage.includes('климат')) {
        return 'Климаттық ақпарат және ұсыныстар:\n\n🌡️ Орташа температура деректері қолжетімді\n🌧️ Жауын-шашын болжамдары\n☀️ Вегетациялық кезең талдауы\n\nҚазіргі климаттық жағдайларды ескере отырып, дақылдар мен егіс мерзімдерін жоспарлауға болады.';
    }
    
    // General help
    if (lowerMessage.includes('көмек') || lowerMessage.includes('анықтама')) {
        return 'Мен сізге мына мәселелер бойынша көмектесе аламын:\n\n📊 Субсидия статистикасы\n🔍 Өтінім верификациясы\n🌾 Топырақ талдауы\n📈 Болжамдар мен трендтер\n🗺️ Облыстық деректер\n💡 AI ұсыныстары\n\nНақты сұрақ қойыңыз немесе төмендегі жиі қойылатын сұрақтардан таңдаңыз!';
    }
    
    // Default response
    return 'Сіздің сұрағыңызды түсіндім. Мен AgroSmart AI көмекшісімін және субсидиялар, жер талдауы, болжамдар және басқа да ауыл шаруашылығы мәселелері бойынша көмектесе аламын.\n\nНақтырақ сұрақ қойыңыз немесе жоғарыдағы жиі қойылатын сұрақтарды пайдаланыңыз.';
}

// ============================================
// Chat History Management
// ============================================

function getChatHistory() {
    return chatHistory;
}

function clearChatHistory() {
    chatHistory = [];
    const chatMessages = document.getElementById('chatMessages');
    
    // Keep only the initial AI message
    const messages = chatMessages.querySelectorAll('.message');
    messages.forEach((msg, index) => {
        if (index > 0) {
            msg.remove();
        }
    });
}

function exportChatHistory() {
    const data = {
        timestamp: new Date().toISOString(),
        messages: chatHistory
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-history-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// ============================================
// Quick Actions Enhancement
// ============================================

function updateQuickActions(newActions) {
    const quickActionsContainer = document.querySelector('.quick-actions');
    if (!quickActionsContainer) return;
    
    const buttonsHTML = newActions.map(action => `
        <button class="quick-action-btn" onclick="sendChatMessage('${action.question}', true)">
            <i class="fas fa-${action.icon}"></i>
            ${action.question}
        </button>
    `).join('');
    
    quickActionsContainer.innerHTML = `
        <h4>Жиі қойылатын сұрақтар:</h4>
        ${buttonsHTML}
    `;
}

// ============================================
// Export Functions
// ============================================

window.ChatModule = {
    sendChatMessage,
    getChatHistory,
    clearChatHistory,
    exportChatHistory,
    updateQuickActions
};

// Make sendChatMessage globally available for quick actions
window.sendChatMessage = sendChatMessage;

console.log('Chat module loaded');
