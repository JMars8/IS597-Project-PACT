import { defineStore } from "pinia";

const PII_KEYS = [
  "identity",
  "location",
  "finance",
  "nationality",
  "medical",
];

export const useUserStore = defineStore("user", {
  state: () => ({
    name: null,
    age: null,
    ethnicity: null,
    // PII onboarding: hide from LLM (true = hide)
    piiPreferences: {
      identity: false,
      location: false,
      finance: false,
      nationality: false,
      medical: false,
    },
    piiOnboardingComplete: false,
  }),
  actions: {
    setUserInfo(name, age, ethnicity) {
      this.name = name;
      this.age = age;
      this.ethnicity = ethnicity;
    },
    setPiiPreference(key, value) {
      if (PII_KEYS.includes(key)) {
        this.piiPreferences[key] = !!value;
      }
    },
    setPiiOnboardingComplete() {
      this.piiOnboardingComplete = true;
    },
  },
});
