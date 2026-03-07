// src/mock/index.js
import Mock from "mockjs";

console.log("%cMock API Enabled", "color: #4ade80");

// Intercept your chat LLM API request
Mock.mock("http://localhost:5173/api/getLLMResponse", "post", (options) => {
  console.log("mock intercepted -> /api/analyze", options);

  const body = JSON.parse(options.body);
  const message = body.message || "";

  // Simple keyword detection mock
  const bullyingKeywords = ["stupid", "ugly", "dumb", "loser"];

  const isBullying = bullyingKeywords.some((w) =>
    message.toLowerCase().includes(w)
  );

  if (isBullying) {
    return {
      flag: 1,
      response:
        "Messages like this can hurt others. Are you sure you want to post? You can check the following resources to get to know more about cyberbullying and its impact on others:\n" +
        "\n• National Mental Health Hotline : 988" +
        "\n• Cyberbullying Resources :" +
        "\n  • https://www.stopbullying.gov/cyberbullying/what-is-it" +
        "\n  • https://www.stopbullying.gov/prevention/how-to-prevent-bullying" +
        "\n  • https://www.stompoutbullying.org/wellness-rooms" +
        "\n  • https://www.unicef.org/stories/how-to-stop-cyberbullying",
    };
  }

  return {
    flag: -1,
    response: null,
  };
});
