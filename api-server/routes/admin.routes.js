const express = require("express");
const router = express.Router();
const { getAllKeys, getStats } = require("../controllers/admin.controller");
const { requireAdmin } = require("../middleware/auth.middleware");

router.get("/keys", requireAdmin, getAllKeys);
router.get("/stats", requireAdmin, getStats);

module.exports = router;
