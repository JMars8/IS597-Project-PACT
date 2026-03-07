import { createApp } from "vue";
import { createPinia } from "pinia";
import router from "./router";
import App from "./App.vue";
import "./mock/index";

const app = createApp(App);
app.use(createPinia());
app.use(router);
app.mount("#app");
