//apiSwitch.js
import { getLLMResponse } from "../services/api";

export const apiSwitch = async (apiKey, req) => {
  let res = {};
  switch (apiKey) {
    case "getLLMResponse":
      res = await getLLMResponse(req);
      break;
    default:
  }
  return res;
};
