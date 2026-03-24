const express = require("express");
const router = express.Router();
const { analyse, status } = require("../controllers/detect.controller");
const { protect } = require("../middleware/auth.middleware");
const { rateLimit } = require("../middleware/ratelimit.middleware");
const upload = require("../middleware/upload.middleware");

router.get("/status", protect, rateLimit, status);
router.post("/analyse", protect, rateLimit, upload, analyse);
router.post("/analyse/gradcam", protect, rateLimit, upload, analyse);

module.exports = router;
