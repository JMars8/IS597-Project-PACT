<template>
  <div class="landing">
    <div class="card">
      <h1 class="title">Welcome</h1>

      <form @submit.prevent="startChat" class="form">
        <label class="label">Name</label>
        <input v-model="name" type="text" required class="input" />

        <label class="label">Age</label>
        <input v-model="age" type="number" required class="input" />

        <label class="label">Ethnicity</label>
        <select v-model="ethnicity" required class="input">
          <option disabled value="">Please select...</option>

          <option value="1">African / African Diaspora</option>
          <option value="2">
            East Asian (e.g., Chinese, Japanese, Korean)
          </option>
          <option value="3">
            South Asian (e.g., Indian, Pakistani, Bangladeshi)
          </option>
          <option value="4">
            Southeast Asian (e.g., Filipino, Vietnamese, Thai, Malay)
          </option>
          <option value="5">Middle Eastern or North African</option>
          <option value="6">Latino / Hispanic</option>
          <option value="7">Indigenous / Native Peoples / First Nations</option>
          <option value="8">European / Caucasian / White</option>
          <option value="9">Pacific Islander</option>
          <option value="10">Mixed Heritage</option>
          <option value="11">Prefer not to say</option>
        </select>

        <button type="submit" class="start-btn">Start Chat</button>
      </form>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { useRouter } from "vue-router";
import { useUserStore } from "../stores/user";

const name = ref("");
const age = ref("");
const ethnicity = ref("");
const router = useRouter();
const userStore = useUserStore();

function startChat() {
  userStore.setUserInfo(name.value, age.value, ethnicity.value);
  router.push("/chat");
}
</script>
<style scoped>
.landing {
  height: 100vh;
  width: 100%;
  background: #f4f3fb;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
    sans-serif;
}
.card {
  background: white;
  padding: 40px 50px;
  border-radius: 24px;
  box-shadow: 0 18px 40px rgba(15, 23, 42, 0.1);
  max-width: 420px;
  width: 100%;
}
.title {
  text-align: center;
  font-size: 1.8rem;
  color: #111827;
  margin-bottom: 28px;
}
.form {
  display: flex;
  flex-direction: column;
  gap: 18px;
}
.label {
  font-size: 0.9rem;
  color: #374151;
  font-weight: 600;
}
.input {
  padding: 12px 16px;
  border: 1px solid #e5d9ff;
  background: #f9fafb;
  border-radius: 12px;
  font-size: 1rem;
  outline: none;
  width: 100%;
  box-sizing: border-box;
}
.input:focus {
  border-color: #a855f7;
  box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.25);
}
.start-btn {
  margin-top: 10px;
  padding: 12px 16px;
  background: #a855f7;
  color: white;
  border-radius: 20px;
  border: none;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: 0.2s ease;
  box-shadow: 0 8px 18px rgba(88, 28, 135, 0.3);
}
.start-btn:hover {
  transform: translateY(-2px);
}
</style>
