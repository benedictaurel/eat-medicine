import express from "express";
import multer from "multer";
import * as tf from "@tensorflow/tfjs";
import * as posenet from "@tensorflow-models/posenet";
import { createCanvas, loadImage } from "canvas";
import fs from "fs";

const app = express();
const upload = multer({ dest: "uploads/" });

let net;

// Load PoseNet model when server starts
(async () => {
  console.log("Loading PoseNet model...");
  net = await posenet.load({
    architecture: "MobileNetV1",
    outputStride: 16,
    inputResolution: { width: 257, height: 200 },
    multiplier: 0.75,
  });
  console.log("✅ PoseNet model loaded!");
})();

// Helper to calculate Euclidean distance
function euclideanDistance(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

// Main endpoint
app.post("/pose", upload.single("image"), async (req, res) => {
  try {
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
      ? euclideanDistance(kp.leftWrist, kp.nose)
      : Infinity;
    const distRight = kp.rightWrist
      ? euclideanDistance(kp.rightWrist, kp.nose)
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
    res.status(500).json({ error: "Pose estimation failed" });
  }
});

// Run the server
app.listen(3000, () =>
  console.log("🚀 PoseNet API running at http://localhost:3000")
);
