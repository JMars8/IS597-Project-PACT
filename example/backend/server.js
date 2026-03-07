import express from "express";
import cors from "cors";
import { spawn } from "child_process";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.join(__dirname, "..");
const chatServerPath = path.join(projectRoot, "LLM", "chat_server.py");

const app = express();
app.use(cors());
app.use(express.json());

let child = null;
let stdoutBuffer = "";
let readyPromise = null;
let nextLineResolve = null;
let nextLinePromise = null;

function resetNextLinePromise() {
  nextLinePromise = new Promise((resolve) => {
    nextLineResolve = resolve;
  });
}

function onStdout(data) {
  stdoutBuffer += data.toString();
  const idx = stdoutBuffer.indexOf("\n");
  if (idx !== -1) {
    const line = stdoutBuffer.slice(0, idx);
    stdoutBuffer = stdoutBuffer.slice(idx + 1);
    if (nextLineResolve) {
      nextLineResolve(line);
      nextLineResolve = null;
    }
  }
}

function getChild() {
  if (child && !child.killed) return Promise.resolve(child);
  if (readyPromise) return readyPromise;

  readyPromise = new Promise((resolve, reject) => {
    const py = spawn("python", [chatServerPath], {
      cwd: projectRoot,
      env: process.env,
      stdio: ["pipe", "pipe", "pipe"],
    });

    stdoutBuffer = "";
    resetNextLinePromise();

    py.stdout.on("data", onStdout);
    py.stderr.on("data", (d) => console.error("[LLM stderr]", d.toString()));

    py.on("error", (err) => {
      console.error("Failed to start Python:", err);
      reject(err);
    });

    py.on("exit", (code, signal) => {
      child = null;
      readyPromise = null;
      if (nextLineResolve) nextLineResolve(null);
    });

    // Wait for READY line
    nextLinePromise.then((line) => {
      if (line === "READY") {
        child = py;
        resolve(py);
      } else {
        reject(new Error("LLM server did not send READY"));
      }
    });
  });

  return readyPromise;
}

function sendRequest(py, message) {
  resetNextLinePromise();
  py.stdin.write(JSON.stringify({ message }) + "\n");
  return nextLinePromise;
}

let requestLock = Promise.resolve();

app.post("/api/getLLMResponse", async (req, res) => {
  const { message } = req.body || {};
  if (!message || typeof message !== "string") {
    return res.status(400).json({ error: "Missing or invalid message" });
  }

  requestLock = requestLock
    .then(async () => {
      let py;
      try {
        py = await getChild();
      } catch (err) {
        return res.status(500).json({
          error: "Failed to start LLM",
          detail: err.message,
        });
      }

      const line = await sendRequest(py, message.trim());
      if (line == null) {
        return res.status(500).json({
          error: "LLM process exited",
          detail: "The model process stopped. Send another message to restart.",
        });
      }

      let out;
      try {
        out = JSON.parse(line);
      } catch (e) {
        return res.status(500).json({
          error: "Invalid response from LLM",
          detail: line,
        });
      }

      if (out.error) {
        return res.status(500).json({
          error: "LLM error",
          detail: out.error,
        });
      }

      return res.json({
        flag: 1,
        response: out.response ?? "",
        feedback_prompt: null,
      });
    })
    .catch((err) => {
      console.error(err);
      res.status(500).json({
        error: "Failed to get LLM response",
        detail: err.message,
      });
    });
});

const port = process.env.PORT || 3001;
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});
