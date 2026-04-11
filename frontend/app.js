const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const batchFromFileBtn = document.getElementById('batch-from-file-btn');
const attachDocBtn = document.getElementById('attach-doc-btn');
const fileInput = document.getElementById('file-input');
const attachmentPreview = document.getElementById('attachment-preview');
const attachedFilename = document.getElementById('attached-filename');
const removeAttachmentBtn = document.getElementById('remove-attachment');
const apiKeyInput = document.getElementById('api-key-input');
const testKeyBtn = document.getElementById('test-key-btn');
const keyStatus = document.getElementById('key-status');

// Load saved API key from session storage
if (apiKeyInput) {
    const savedKey = sessionStorage.getItem('pact_openai_key');
    if (savedKey) {
        apiKeyInput.value = savedKey;
        updateKeyStatus('active', '✅');
        console.log("DEBUG: Restored API key from session storage.");
    }
}
const llamaStatusBadge = document.getElementById('llama-status-badge');
const llamaStatusDot = document.getElementById('llama-status-dot');
const llamaStatusText = document.getElementById('llama-status-text');

let currentAttachmentText = "";
let currentAttachmentName = "";

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

async function sendMessage() {
    let text = userInput.value.trim();
    if (!text && !currentAttachmentText) return;

    // Combine document and question if attachment exists
    let fullQuery = text;
    if (currentAttachmentText) {
        fullQuery = `[Document Content: ${currentAttachmentName}]\n${currentAttachmentText}\n\n[User Question]: ${text}`;
    }

    // Add user message to UI (show a shorter version if it's a huge doc)
    const displayMessage = currentAttachmentName ? `📎 ${currentAttachmentName}\n${text}` : text;
    appendMessage('user', displayMessage);
    userInput.value = '';
    userInput.style.height = 'auto';

    // Show typing indicator or placeholder
    const botMsgId = appendMessage('bot', 'Sanitizing and thinking...');

    try {
        // Prepare request body
        const keyEl = document.getElementById('api-key-input');
        let userApiKey = keyEl ? keyEl.value.trim() : null;
        
        // Fallback to session storage if the input exists but is empty
        if (!userApiKey) {
            userApiKey = sessionStorage.getItem('pact_openai_key');
        }
        
        const payload = {
            query: fullQuery,
            is_document: !!currentAttachmentText,
            api_key: userApiKey,
            settings: {
                identity: toggles.identity.checked,
                location: toggles.location.checked,
                demographic: toggles.demographic.checked,
                health: toggles.health.checked,
                financial: toggles.financial.checked
            }
        };

        console.log("DEBUG: Sending payload to /chat", {
            query_length: payload.query.length,
            has_api_key: !!payload.api_key,
            api_key_length: payload.api_key ? payload.api_key.length : 0
        });

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
            data.sanitizations || [],
            data.pipeline_trace || null
        );

        // Clear attachment after successful send
        clearAttachment();

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
        const keyEl = document.getElementById('api-key-input');
        let userApiKey = keyEl ? keyEl.value.trim() : null;
        if (!userApiKey) {
            userApiKey = sessionStorage.getItem('pact_openai_key');
        }

        const response = await fetch('http://localhost:8000/chat/batch-from-file', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                api_key: userApiKey
            })
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

attachDocBtn.addEventListener('click', () => fileInput.click());

// API Key Helpers
function updateKeyStatus(mode, icon) {
    const keyStatus = document.getElementById('key-status');
    if (!keyStatus) return;
    if (mode === 'active') {
        keyStatus.textContent = icon + ' active';
        keyStatus.style.color = '#22d3ee'; // var(--accent-cyan)
        keyStatus.style.fontWeight = 'bold';
    } else if (mode === 'error') {
        keyStatus.textContent = icon + ' invalid';
        keyStatus.style.color = '#ff6b6b';
        keyStatus.style.fontWeight = 'bold';
    } else {
        keyStatus.textContent = '';
    }
}

if (apiKeyInput) {
    apiKeyInput.addEventListener('input', () => {
        const val = apiKeyInput.value.trim();
        if (val) {
            sessionStorage.setItem('pact_openai_key', val);
            updateKeyStatus('active', '•');
        } else {
            sessionStorage.removeItem('pact_openai_key');
            updateKeyStatus('empty', '');
        }
    });

    apiKeyInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            const val = apiKeyInput.value.trim();
            if (val) {
                updateKeyStatus('active', '✅');
                apiKeyInput.blur();
            }
        }
    });
}

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const originalPlaceholder = userInput.placeholder;
    userInput.placeholder = "Reading attachment...";
    userInput.disabled = true;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://localhost:8000/extract/text', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!response.ok) {
            alert(data.detail || "Extraction failed");
            return;
        }

        // Store internally, don't show all text in box
        currentAttachmentText = data.text;
        currentAttachmentName = file.name;
        
        // Show the badge
        attachedFilename.textContent = `Attached: ${file.name}`;
        attachmentPreview.style.display = 'block';
        userInput.placeholder = "Ask a question about this attachment...";
        userInput.focus();
        
    } catch (error) {
        console.error('Extraction Error:', error);
        alert("Failed to extract document text.");
    } finally {
        userInput.disabled = false;
        fileInput.value = '';
    }
});

function clearAttachment() {
    currentAttachmentText = "";
    currentAttachmentName = "";
    attachmentPreview.style.display = 'none';
    userInput.placeholder = "Type your message here...";
}

removeAttachmentBtn.addEventListener('click', clearAttachment);
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

    const setText = (t, status = 'loading') => {
        const el = document.getElementById(msgId);
        if (el) el.textContent = t;
        
        if (status === 'ready') {
            setHeaderStatus('ready', 'Llama 3 Local Sanitizer Active');
        } else if (status === 'error') {
            setHeaderStatus('error', 'Llama 3 Unavailable');
        } else {
            setHeaderStatus('loading', 'Llama 3 Loading...');
        }
    };

    const setHeaderStatus = (status, text) => {
        if (!llamaStatusBadge) return;
        llamaStatusText.textContent = text;
        llamaStatusBadge.style.opacity = "1";
        if (status === 'ready') {
            llamaStatusDot.style.background = "var(--accent-cyan)";
            llamaStatusBadge.style.color = "var(--accent-cyan)";
        } else if (status === 'loading') {
            llamaStatusDot.style.background = "#fbbf24";
            llamaStatusBadge.style.color = "#fbbf24";
        } else {
            llamaStatusDot.style.background = "#ef4444";
            llamaStatusBadge.style.color = "#ef4444";
        }
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
                setText('Local Llama: ready.', 'ready');
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
                setText(`Local Llama: failed to load: ${st.load_error}`, 'error');
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
            setText('Local Llama: status unavailable. (Check backend.)', 'error');
            return;
        }

        await new Promise((r) => setTimeout(r, POLL_MS));
    }

    setText(
        'Local Llama: UI wait limit (24h). Load may still run on the server — hard-refresh (Ctrl+Shift+R) to reload app.js, check GET /local-llama/status, or set LOCAL_LLM_LOAD_TIMEOUT_SEC on the backend.'
    );
})();
