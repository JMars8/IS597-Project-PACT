<template>
  <!-- PII preferences popup (shown on first visit) -->
  <div v-if="!userStore.piiOnboardingComplete" class="popup-overlay">
    <div class="popup-card">
      <h2 class="popup-title">What type of personal information would you like to hide from LLM?</h2>
      <p class="popup-subtitle">Toggle Yes to hide each category from the model.</p>

      <div class="popup-options">
        <div
          v-for="item in piiOptions"
          :key="item.key"
          class="popup-row"
        >
          <span class="popup-label">{{ item.label }}</span>
          <button
            type="button"
            class="toggle-btn"
            :class="{ on: userStore.piiPreferences[item.key] }"
            :aria-pressed="userStore.piiPreferences[item.key]"
            @click="togglePii(item.key)"
          >
            <span class="toggle-knob"></span>
          </button>
          <span class="popup-value">{{ userStore.piiPreferences[item.key] ? 'Yes' : 'No' }}</span>
        </div>
      </div>

      <button type="button" class="popup-continue" @click="continueToApp">
        Continue
      </button>
    </div>
  </div>

  <router-view />
</template>

<script setup>
import { useUserStore } from "./stores/user";

const userStore = useUserStore();

const piiOptions = [
  { key: "identity", label: "1. Identity / Direct PII" },
  { key: "location", label: "2. Location / Geo" },
  { key: "finance", label: "3. Finance" },
  { key: "nationality", label: "4. Nationality / Demographics" },
  { key: "medical", label: "5. Medical / Health" },
];

function togglePii(key) {
  userStore.setPiiPreference(key, !userStore.piiPreferences[key]);
}

function continueToApp() {
  userStore.setPiiOnboardingComplete();
}
</script>

<style>
html,
body,
#app {
  margin: 0;
  padding: 0;
  height: 100%;
  width: 100%;
  background: #f4f3fb;
}

/* PII popup */
.popup-overlay {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
  padding: 16px;
}

.popup-card {
  background: white;
  border-radius: 24px;
  box-shadow: 0 24px 48px rgba(15, 23, 42, 0.2);
  padding: 36px 40px;
  max-width: 480px;
  width: 100%;
}

.popup-title {
  font-size: 1.35rem;
  color: #111827;
  margin: 0 0 8px;
  line-height: 1.4;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.popup-subtitle {
  font-size: 0.9rem;
  color: #6b7280;
  margin: 0 0 24px;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.popup-options {
  display: flex;
  flex-direction: column;
  gap: 14px;
  margin-bottom: 28px;
}

.popup-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.popup-label {
  flex: 1;
  font-size: 0.95rem;
  color: #374151;
  font-weight: 500;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.popup-value {
  font-size: 0.85rem;
  color: #6b7280;
  min-width: 28px;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.toggle-btn {
  width: 48px;
  height: 26px;
  border-radius: 9999px;
  border: 2px solid #e5e7eb;
  background: #e5e7eb;
  cursor: pointer;
  padding: 0;
  flex-shrink: 0;
  transition: background 0.2s, border-color 0.2s;
}

.toggle-btn .toggle-knob {
  display: block;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background: white;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
  margin-left: 0;
  transition: margin-left 0.2s;
}

.toggle-btn.on {
  background: #a855f7;
  border-color: #a855f7;
}

.toggle-btn.on .toggle-knob {
  margin-left: 22px;
}

.popup-continue {
  width: 100%;
  padding: 14px 20px;
  background: #a855f7;
  color: white;
  border: none;
  border-radius: 20px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  box-shadow: 0 8px 18px rgba(88, 28, 135, 0.3);
  transition: transform 0.2s;
}

.popup-continue:hover {
  transform: translateY(-1px);
}
</style>
