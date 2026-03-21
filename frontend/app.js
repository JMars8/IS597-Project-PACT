const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

// Toggles
const toggles = {
    identity: document.getElementById('toggle-identity'),
    location: document.getElementById('toggle-location'),
    demographic: document.getElementById('toggle-demographic'),
    health: document.getElementById('toggle-health'),
    financial: document.getElementById('toggle-financial')
};

// Auto-resize textarea
userInput.addEventListener('input', () => {
    userInput.style.height = 'auto';
    userInput.style.height = (userInput.scrollHeight) + 'px';
});

// Send message logic
async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    // Add user message to UI
    appendMessage('user', text);
    userInput.value = '';
    userInput.style.height = 'auto';

    // Show typing indicator or placeholder
    const botMsgId = appendMessage('bot', 'Sanitizing and thinking...');

    try {
        // Prepare request body
        const payload = {
            query: text,
            settings: {
                identity: toggles.identity.checked,
                location: toggles.location.checked,
                demographic: toggles.demographic.checked,
                health: toggles.health.checked,
                financial: toggles.financial.checked
            }
        };

        const response = await fetch('http://localhost:8000/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();
        updateMessage(botMsgId, data.response, data.sanitizations);

    } catch (error) {
        console.error('Error:', error);
        updateMessage(botMsgId, "Sorry, I couldn't connect to the backend sanitizer. Make sure the Python server is running.");
    }
}

function appendMessage(role, text) {
    const msgDiv = document.createElement('div');
    const id = 'msg-' + Date.now();
    msgDiv.id = id;
    msgDiv.className = `message ${role}-message`;
    msgDiv.textContent = text;
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return id;
}

function updateMessage(id, text, sanitizations = []) {
    const msgDiv = document.getElementById(id);
    if (!msgDiv) return;

    msgDiv.innerHTML = text;

    if (sanitizations && sanitizations.length > 0) {
        const sanitizationList = document.createElement('div');
        sanitizationList.style.marginTop = '8px';
        sanitizationList.style.padding = '8px';
        sanitizationList.style.fontSize = '0.75rem';
        sanitizationList.style.background = 'rgba(0,0,0,0.1)';
        sanitizationList.style.borderRadius = '8px';
        sanitizationList.style.borderLeft = '2px solid var(--accent-cyan)';
        
        let html = '<strong>Privacy Actions:</strong><ul style="margin-left: 15px; margin-top: 4px;">';
        sanitizations.forEach(s => {
            const type = s.type || 'SENSITIVE';
            html += `<li>Redacted ${type} info: <span style="color: var(--accent-cyan);">${s.original}</span> → [REDACTED]</li>`;
        });
        html += '</ul>';
        sanitizationList.innerHTML = html;
        msgDiv.appendChild(sanitizationList);
    }
    
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

sendBtn.addEventListener('click', sendMessage);
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
