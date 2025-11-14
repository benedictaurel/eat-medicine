import express from "express";
import multer from "multer";
import * as tf from "@tensorflow/tfjs";
import * as posenet from "@tensorflow-models/posenet";
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
  }
  return net;
}

// Root endpoint
app.get("/", (req, res) => {
  res.json({
    message: "This is an API for PoseNet pose estimation that identifies the closest hand to the nose in an image to help patients that suffer from tuberculosis."
  });
});

// Main endpoint
app.post("/pose", upload.single("image"), async (req, res) => {
  try {
    // Initialize model if not loaded
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

    // Get keypoints
    const kp = {};
    for (let point of pose.keypoints) {
      if (["nose", "leftWrist", "rightWrist"].includes(point.part)) {
        kp[point.part] = point.position;
      }
    }

    if (!kp.nose) {
      return res.status(400).json({ error: "Nose not detected" });
    }

    const distLeft = kp.leftWrist
      ? dist(kp.leftWrist, kp.nose)
      : Infinity;
    const distRight = kp.rightWrist
      ? dist(kp.rightWrist, kp.nose)
      : Infinity;
    const minDist = Math.min(distLeft, distRight);
    const closestHand = minDist === distLeft ? "leftWrist" : "rightWrist";

    const THRESHOLD_DISTANCE = 550;
    const confidenceDecimal = Math.max(0, 1 - minDist / THRESHOLD_DISTANCE);
    const ACCEPTANCE_THRESHOLD = 0.1;

    const result = {
      closestHand,
      distance: parseFloat(minDist.toFixed(2)),
      accepted: confidenceDecimal >= ACCEPTANCE_THRESHOLD,
    };

    // Clean up uploaded image
    fs.unlinkSync(req.file.path);

    return res.json(result);
  } catch (error) {
    console.error("Pose estimation error:", error);

    // Clean up uploaded file if it exists
    if (req.file && fs.existsSync(req.file.path)) {
      fs.unlinkSync(req.file.path);
    }

    res.status(500).json({ error: "Pose estimation failed", details: error.message });
  }
});

// Export app for Vercel serverless environment
export default app;
