<template>
  <div class="app">
    <!-- SIDEBAR -->
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-logo">🥑</div>
        <div class="brand-text">
          <div class="brand-sub">LLM</div>
        </div>
      </div>
    </aside>

    <!-- MAIN -->
    <main class="main">
      <header class="topbar">
        <button class="new-chat-btn" @click="closeChat">☓ Close Chat</button>
      </header>

      <!-- CHAT AREA -->
      <section class="chat-area" ref="chatWindowRef">
        <div
          v-for="message in messages"
          :key="message.id"
          class="message-row"
          :class="message.role"
        >
          <div class="message-bubble">
            <p
              v-for="(line, i) in message.text.split('\n')"
              :key="i"
              class="message-line"
            >
              {{ line }}
            </p>
          </div>
        </div>
      </section>

      <!-- INPUT -->
      <footer class="composer">
        <p v-if="isWaiting" class="composer-status">PACT is thinking… (first reply may take a minute)</p>
        <div class="composer-box">
          <textarea
            v-model="draft"
            class="composer-input"
            placeholder="Ask me anything . . ."
            rows="1"
            :disabled="isWaiting"
            @keydown.enter.exact.prevent="sendMessage"
          ></textarea>

          <button
            class="composer-send-btn"
            type="button"
            :disabled="!draft.trim() || isWaiting"
            @click="sendMessage"
          >
            ➤
          </button>
        </div>
      </footer>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from "vue";
import { useUserStore } from "../stores/user";
import { apiSwitch } from "../stores/apiSwitch";
import { v4 as uuid } from "uuid";

// Store (name, age, ethnicity)
const userStore = useUserStore();

// Initial greeting
function getInitialMessages() {
  return [
    {
      id: uuid(),
      role: "assistant",
      text: "Hi! I am PACT, your privacy-protecting LLM Agent.",
    },
  ];
}

const messages = ref(getInitialMessages());
const draft = ref("");
const chatWindowRef = ref(null);
const isWaiting = ref(false);

onMounted(() => {
  scrollToBottom();
});

watch(messages, scrollToBottom, { deep: true });

function scrollToBottom() {
  const el = chatWindowRef.value;
  if (!el) return;
  requestAnimationFrame(() => {
    el.scrollTop = el.scrollHeight;
  });
}

function resetChat() {
  messages.value = getInitialMessages();
}

function closeChat() {
  resetChat();
  window.open("https://forms.gle/MUsynVmuZU9bfesN8", "_blank");
}

async function sendMessage() {
  const text = draft.value.trim();
  if (!text) return;

  // Add user message
  messages.value.push({
    id: uuid(),
    role: "user",
    text,
  });

  draft.value = "";
  isWaiting.value = true;

  try {
    const request = {
      age: userStore.age ?? 25,
      ethnicity: userStore.ethnicity ?? "",
      message: text,
    };
    const response = await apiSwitch("getLLMResponse", request);
    const data = response.data;

    // Normal chat: backend returns { flag: 1, response: "..." }
    if (data.flag === 1) {
      messages.value.push({
        id: uuid(),
        role: "assistant",
        text: data.response ?? "",
      });
      if (data.feedback_prompt) {
        messages.value.push({
          id: uuid(),
          role: "assistant",
          text: data.feedback_prompt,
        });
      }
    }
  } catch (err) {
    const detail = err.response?.data?.detail ?? err.response?.data?.error ?? err.message;
    messages.value.push({
      id: uuid(),
      role: "assistant",
      text: "⚠️ Sorry, something went wrong.\n\n" + detail,
    });
  } finally {
    isWaiting.value = false;
  }
}

// ========================================================================================================================
</script>

<style scoped>
/* Layout */
.app {
  display: flex;
  height: 100vh;
  height: auto;
  width: 100%;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
    sans-serif;
}

/* Sidebar */
.sidebar {
  width: 260px;
  background: #111827;
  color: #e5e7eb;
  padding: 24px 20px;
  box-sizing: border-box;
  display: flex;
  flex-direction: column;
  min-height: 100vh; /* FIX: ensures full height even when content scrolls */
}

.brand {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 32px;
}

.brand-logo {
  width: 40px;
  height: 40px;
  border-radius: 9999px;
  background: linear-gradient(135deg, #8b5cf6, #ec4899);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 0.95rem;
  color: #f9fafb;
}

.brand-text {
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 0.9rem;
}

.brand-sub {
  color: #a5b4fc;
  font-weight: 700;
}

/* Main area */
.main {
  flex: 1;
  background: #f4f3fb;
  display: flex;
  flex-direction: column;
  margin: 0 10%;
}

/* Top bar */
.topbar {
  display: flex;
  justify-content: flex-end;
  padding: 24px 5px;
}

/* New chat button */
.new-chat-btn {
  border: none;
  background: #000000;
  color: #f9fafb;
  padding: 1.5% 20px;
  border-radius: 20px;
  font-size: 0.9rem;
  font-weight: 500;
  cursor: pointer;
  box-shadow: 0 8px 18px rgba(88, 28, 135, 0.3);
}

/* Chat area */
.chat-area {
  flex: 1;
  padding: 24px 80px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

/* Messages */
.message-row {
  max-width: 640px;
}

.message-row.user {
  align-self: flex-end;
}

.message-row.assistant {
  align-self: flex-start;
}

/* user bubble */
.message-row.user .message-bubble {
  background: #a855f7;
  color: #f9fafb;
  padding: 1.5% 20px;
  border-radius: 20px;
  font-size: 0.9rem;
}

/* assistant card */
.message-row.assistant .message-bubble {
  background: #ffffff;
  color: #111827;
  padding: 1.5% 20px;
  border-radius: 20px;
  font-size: 0.9rem;
  line-height: 1.6;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.08);
}

.message-line + .message-line {
  margin-top: 4px;
}

/* Composer */
.composer {
  padding: 0 80px 32px;
}

.composer-status {
  font-size: 0.85rem;
  color: #6b7280;
  margin: 0 0 8px;
}

.composer-box {
  width: 100%;
  max-width: 720px;
  margin: 0 auto;
  background: #f9fafb;
  border-radius: 9999px;
  border: 1px solid #e5d9ff;
  display: flex;
  align-items: center;
  padding: 8px 14px;
  box-shadow: 0 16px 30px rgba(148, 163, 184, 0.25);
  gap: 8px;
}

.composer-input {
  flex: 1;
  border: none;
  background: transparent;
  resize: none;
  outline: none;
  font-size: 0.95rem;
  color: #111827;
}

.composer-input::placeholder {
  color: #9ca3af;
}

.composer-send-btn {
  width: 32px;
  height: 32px;
  border-radius: 9999px;
  border: none;
  background: #a855f7;
  color: #fff;
  cursor: pointer;
}
</style>
