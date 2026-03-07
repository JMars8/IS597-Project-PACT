import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import { spawn } from "child_process";

const app = express();
app.use(cors());
app.use(bodyParser.json());

// -------------------------
// POST /api/analyze
// -------------------------
app.post("/api/getLLMResponse", (req, res) => {
  const { message, age, ethnicity } = req.body;

  // Call your Python classifier
  const py = spawn("python3", ["model/run_once.py", message, age, ethnicity]);

  let result = "";

  py.stdout.on("data", (data) => {
    result += data.toString();
  });

  py.stderr.on("data", (data) => {
    console.error("Python error:", data.toString());
  });

  py.on("close", () => {
    try {
      const output = JSON.parse(result);
      res.json(output);
    } catch (e) {
      res.status(500).json({ error: "Invalid JSON from Python", raw: result });
    }
  });
});

app.listen(3001, () => {
  console.log("Backend running on http://localhost:3001");
});
