const KeyStore = require("../utils/keyStore");

const protect = (req, res, next) => {
  const key = req.headers["x-api-key"];
  const keys = KeyStore.load();

  if (!key) {
    return res.status(401).json({
      success: false,
      error: "missing_api_key",
      message: "Include your API key in the X-API-Key header.",
    });
  }
  if (!keys[key]) {
    return res.status(401).json({
      success: false,
      error: "invalid_api_key",
      message: "API key not recognised.",
    });
  }
  if (!keys[key].active) {
    return res.status(403).json({
      success: false,
      error: "key_disabled",
      message: "This API key has been disabled.",
    });
  }

  req.apiKey = key;
  req.keyMeta = keys[key];
  next();
};

const requireAdmin = (req, res, next) => {
  const key = req.headers["x-api-key"];
  const keys = KeyStore.load();

  if (!key || !keys[key] || !keys[key].admin) {
    return res.status(403).json({
      success: false,
      error: "admin_required",
      message: "Admin API key required.",
    });
  }

  req.apiKey = key;
  req.keyMeta = keys[key];
  next();
};

module.exports = { protect, requireAdmin };
