const express = require("express");
const router = express.Router();
const {
  register,
  createKey,
  revokeKey,
} = require("../controllers/auth.controller");
const { requireAdmin } = require("../middleware/auth.middleware");

router.post("/register", register);
router.post("/keys", requireAdmin, createKey);
router.delete("/keys/:prefix", requireAdmin, revokeKey);

module.exports = router;