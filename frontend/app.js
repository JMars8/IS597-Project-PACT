const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const batchFromFileBtn = document.getElementById('batch-from-file-btn');

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

        const data = await response.json().catch(() => null);
        if (!response.ok) {
            const detail = data && data.detail ? data.detail : `Request failed with status ${response.status}`;
            updateMessage(botMsgId, escapeHtml(String(detail)), []);
            return;
        }

        updateMessage(
            botMsgId,
            data.response,
            data.sanitizations,
            data.pipeline_trace || null
        );

    } catch (error) {
        console.error('Error:', error);
        updateMessage(
            botMsgId,
            "Sorry, I couldn't connect to the backend sanitizer. Make sure the Python server is running.",
            []
        );
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

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function appendPipelineTrace(container, trace) {
    if (!trace || typeof trace !== 'object') return;

    const wrap = document.createElement('details');
    wrap.style.marginTop = '12px';
    wrap.style.border = '1px solid var(--glass-border, rgba(255,255,255,0.12))';
    wrap.style.borderRadius = '8px';
    wrap.style.padding = '8px 10px';
    wrap.style.background = 'rgba(0,0,0,0.15)';

    const sum = document.createElement('summary');
    sum.style.cursor = 'pointer';
    sum.style.fontSize = '0.8rem';
    sum.style.fontWeight = '600';
    sum.style.color = 'var(--accent-cyan, #5eead4)';
    sum.textContent = 'Pipeline: module masks → Local Llama → GPT';

    const pre = document.createElement('pre');
    pre.style.margin = '8px 0 0 0';
    pre.style.whiteSpace = 'pre-wrap';
    pre.style.wordBreak = 'break-word';
    pre.style.fontSize = '0.68rem';
    pre.style.lineHeight = '1.35';
    pre.style.maxHeight = 'min(50vh, 420px)';
    pre.style.overflow = 'auto';
    pre.textContent = JSON.stringify(trace, null, 2);

    wrap.appendChild(sum);
    wrap.appendChild(pre);
    container.appendChild(wrap);
}

function appendSanitizationsBlock(container, sanitizations) {
    if (!sanitizations || sanitizations.length === 0) return;
    const sanitizationList = document.createElement('div');
    sanitizationList.style.marginTop = '8px';
    sanitizationList.style.padding = '8px';
    sanitizationList.style.fontSize = '0.75rem';
    sanitizationList.style.background = 'rgba(0,0,0,0.1)';
    sanitizationList.style.borderRadius = '8px';
    sanitizationList.style.borderLeft = '2px solid var(--accent-cyan)';

    let html = '<strong>Privacy Actions:</strong><ul style="margin-left: 15px; margin-top: 4px;">';
    sanitizations.forEach((s) => {
        const orig = escapeHtml(s.original);
        html += `<li>Redacted financial info: <span style="color: var(--accent-cyan);">${orig}</span> → [REDACTED]</li>`;
    });
    html += '</ul>';
    sanitizationList.innerHTML = html;
    container.appendChild(sanitizationList);
}

function updateMessage(id, text, sanitizations = [], pipelineTrace = null) {
    const msgDiv = document.getElementById(id);
    if (!msgDiv) return;

    msgDiv.innerHTML = '';
    appendPipelineTrace(msgDiv, pipelineTrace);
    const body = document.createElement('div');
    body.className = 'bot-reply-body';
    body.innerHTML = text;
    msgDiv.appendChild(body);
    appendSanitizationsBlock(msgDiv, sanitizations);

    chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function runBatchFromFile() {
    appendMessage('user', '[Run batch: data/queries.json]');
    const botMsgId = appendMessage('bot', 'Loading queries from JSON and sanitizing…');

    try {
        const response = await fetch('http://localhost:8000/chat/batch-from-file', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
        });
        const data = await response.json().catch(() => null);
        if (!response.ok) {
            const detail = data && data.detail ? data.detail : `Request failed with status ${response.status}`;
            updateMessage(botMsgId, escapeHtml(String(detail)), []);
            return;
        }

        const msgDiv = document.getElementById(botMsgId);
        if (!msgDiv) return;

        msgDiv.innerHTML = '';
        data.results.forEach((r, i) => {
            const block = document.createElement('div');
            block.className = 'batch-result';
            block.style.marginBottom = '1.25rem';
            block.style.paddingBottom = '1rem';
            block.style.borderBottom =
                i < data.results.length - 1 ? '1px solid var(--glass-border)' : 'none';

            const label = document.createElement('div');
            label.style.fontSize = '0.75rem';
            label.style.color = 'var(--text-muted)';
            label.style.marginBottom = '6px';
            label.textContent = `Query ${i + 1} of ${data.count}`;
            block.appendChild(label);

            appendPipelineTrace(block, r.pipeline_trace || null);

            const body = document.createElement('div');
            body.innerHTML = r.response;
            block.appendChild(body);
            appendSanitizationsBlock(block, r.sanitizations);
            msgDiv.appendChild(block);
        });

        chatMessages.scrollTop = chatMessages.scrollHeight;
    } catch (error) {
        console.error('Batch error:', error);
        updateMessage(
            botMsgId,
            'Could not run data/queries.json. Is the backend running and is the file present on the server?',
            []
        );
    }
}

sendBtn.addEventListener('click', sendMessage);
batchFromFileBtn.addEventListener('click', runBatchFromFile);
userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Preload local Llama on page load so the user sees progress immediately.
// First-time HF download + 8B load can take hours on a slow link; polling runs up to 24h.
(async function bootLocalLlama() {
    if (!chatMessages) return;

    const POLL_MS = 2500;
    // Match long server wait (default 6h); cap UI polling at 24h so tabs don't run forever.
    const LOAD_DEADLINE_MS = Date.now() + 24 * 60 * 60 * 1000;
    const started = Date.now();

    const msgId = appendMessage('bot', 'Local Llama: loading…');

    const setText = (t) => {
        const el = document.getElementById(msgId);
        if (el) el.textContent = t;
    };

    const fmtElapsed = () => {
        const m = Math.floor((Date.now() - started) / 60000);
        const s = Math.floor(((Date.now() - started) % 60000) / 1000);
        return m > 0 ? `${m}m ${s}s` : `${s}s`;
    };

    while (Date.now() < LOAD_DEADLINE_MS) {
        try {
            const statusResp = await fetch('http://localhost:8000/local-llama/status');
            const st = await statusResp.json().catch(() => ({}));

            if (st.loaded) {
                setText('Local Llama: ready.');
                return;
            }

            if (st.loading) {
                setText(
                    `Local Llama: loading… (${fmtElapsed()} elapsed, ${st.model_name || 'model'})`
                );
                await new Promise((r) => setTimeout(r, POLL_MS));
                continue;
            }

            if (st.load_error) {
                setText(`Local Llama: failed to load: ${st.load_error}`);
                return;
            }

            // Not loaded and not currently loading -> kick off loading.
            await fetch('http://localhost:8000/local-llama/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({}),
            }).catch(() => {});

            setText(`Local Llama: starting load… (${fmtElapsed()} elapsed)`);
        } catch (e) {
            console.error('Local Llama status error:', e);
            setText('Local Llama: status unavailable. (Check backend.)');
            return;
        }

        await new Promise((r) => setTimeout(r, POLL_MS));
    }

    setText(
        'Local Llama: UI wait limit (24h). Load may still run on the server — hard-refresh (Ctrl+Shift+R) to reload app.js, check GET /local-llama/status, or set LOCAL_LLM_LOAD_TIMEOUT_SEC on the backend.'
    );
})();
