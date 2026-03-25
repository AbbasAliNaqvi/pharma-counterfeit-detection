const ort = require("onnxruntime-web");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

const MODEL_PATH = process.env.MODEL_PATH
  ? path.resolve(process.env.MODEL_PATH)
  : path.join(__dirname, "../models/pharmaguard.onnx");
const IMG_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const CLASSES = { 0: "Authentic", 1: "Counterfeit" };

let _session = null;

async function getSession() {
  if (!fs.existsSync(MODEL_PATH)) {
    throw new Error(`Model file not found at ${MODEL_PATH}`);
  }

  if (!_session) {
    _session = await ort.InferenceSession.create(MODEL_PATH);
  }
  return _session;
}

async function preprocessImage(buffer) {
  const raw = await sharp(buffer)
    .resize(IMG_SIZE, IMG_SIZE)
    .removeAlpha()
    .raw()
    .toBuffer();

  const float32 = new Float32Array(3 * IMG_SIZE * IMG_SIZE);

  for (let i = 0; i < IMG_SIZE * IMG_SIZE; i++) {
    float32[i] = (raw[i * 3] / 255 - MEAN[0]) / STD[0];
    float32[IMG_SIZE * IMG_SIZE + i] =
      (raw[i * 3 + 1] / 255 - MEAN[1]) / STD[1];
    float32[2 * IMG_SIZE * IMG_SIZE + i] =
      (raw[i * 3 + 2] / 255 - MEAN[2]) / STD[2];
  }

  return new ort.Tensor("float32", float32, [1, 3, IMG_SIZE, IMG_SIZE]);
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

function riskLevel(fakeProbability) {
  if (fakeProbability < 0.2) return "LOW";
  if (fakeProbability < 0.5) return "MEDIUM";
  if (fakeProbability < 0.8) return "HIGH";
  return "CRITICAL";
}

async function runInference(imageBuffer) {
  const session = await getSession();
  const tensor = await preprocessImage(imageBuffer);
  const feeds = { image: tensor };
  const results = await session.run(feeds);
  const logits = Array.from(results.logits.data);
  const probs = softmax(logits);
  const predIdx = probs[0] > probs[1] ? 0 : 1;

  return {
    prediction: CLASSES[predIdx],
    class_index: predIdx,
    is_authentic: predIdx === 0,
    confidence: parseFloat(probs[predIdx].toFixed(6)),
    probabilities: {
      authentic: parseFloat(probs[0].toFixed(6)),
      counterfeit: parseFloat(probs[1].toFixed(6)),
    },
    risk_level: riskLevel(probs[1]),
  };
}

module.exports = { runInference };
