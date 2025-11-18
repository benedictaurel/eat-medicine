import express from "express";
import multer from "multer";
import * as tf from "@tensorflow/tfjs-node";
import posenet from "@tensorflow-models/posenet";
import { createCanvas, loadImage } from "canvas";
import fs from "fs";
import path from "path";
import os from "os";

const app = express();
const upload = multer({ dest: path.join(os.tmpdir(), "uploads") });

let net;

// Helper to calculate Euclidean distance
function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

// Initialize PoseNet model
async function initModel() {
  if (!net) {
    console.log("Loading PoseNet model...");
    net = await posenet.load({
      architecture: "MobileNetV1",
      outputStride: 16,
      inputResolution: { width: 257, height: 200 },
      multiplier: 0.75,
    });
    console.log("✅ PoseNet model loaded!");

    // Warm up the model once
    const warmupInput = tf.zeros([1, 200, 257, 3]);
    await net.estimateSinglePose(warmupInput);
    warmupInput.dispose();
  }
  return net;
}

app.get("/", (req, res) => {
  res.json({
    message:
      "PoseNet API using TensorFlow.js Node — detects closest hand to the nose.",
  });
});

app.post("/pose", upload.single("image"), async (req, res) => {
  try {
    await initModel();

    if (!req.file) {
      return res.status(400).json({ error: "No image uploaded" });
    }

    const image = await loadImage(req.file.path);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, 0, 0);

    // Estimate pose
    const pose = await net.estimateSinglePose(canvas, {
      flipHorizontal: false,
    });

    const kp = {};
    for (let point of pose.keypoints) {
      if (point.score < 0.5) continue;
      if (["nose", "leftWrist", "rightWrist"].includes(point.part)) {
        kp[point.part] = point.position;
      }
    }

    console.log("Detected keypoints:", kp);

    if (!kp.nose) {
      return res.status(400).json({ error: "Nose not detected", accepted: false });
    }

    if (!kp.rightWrist && !kp.leftWrist) {
      return res.status(400).json({ error: "No wrist detected", accepted: false });
    }

    const nearestHand = kp.rightWrist || kp.leftWrist;
    const d = dist(nearestHand, kp.nose);

    const THRESHOLD_DISTANCE = 1500;
    const confidence = Math.max(0, 1 - d / THRESHOLD_DISTANCE);
    const ACCEPTANCE_THRESHOLD = 0.1;

    const result = {
      distance: parseFloat(d.toFixed(2)),
      accepted: confidence >= ACCEPTANCE_THRESHOLD,
      confidence: parseFloat(confidence.toFixed(3)),
    };

    fs.unlinkSync(req.file.path);
    return res.json(result);
  } catch (error) {
    console.error("Pose estimation error:", error);

    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }
    res.status(500).json({ error: "Pose estimation failed", details: error.message });
  }
});

// Export for serverless deployments
export default app;
