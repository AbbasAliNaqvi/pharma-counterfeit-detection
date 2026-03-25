const KeyStore = require("../utils/keyStore");

const protect = async (req, res, next) => {
  try {
    const key = req.headers["x-api-key"];

    if (!key) {
      return res.status(401).json({
        success: false,
        error: "missing_api_key",
        message: "Include your API key in the X-API-Key header.",
      });
    }

    const keyDoc = await KeyStore.findByKey(key);

    if (!keyDoc) {
      return res.status(401).json({
        success: false,
        error: "invalid_api_key",
        message: "API key not recognised.",
      });
    }

    if (!keyDoc.active) {
      return res.status(403).json({
        success: false,
        error: "key_disabled",
        message: "This API key has been disabled.",
      });
    }

    req.apiKey = key;
    req.keyMeta = keyDoc;
    next();
  } catch (error) {
    next(error);
  }
};

const requireAdmin = async (req, res, next) => {
  try {
    const key = req.headers["x-api-key"];
    const keyDoc = key ? await KeyStore.findByKey(key) : null;

    if (!key || !keyDoc || !keyDoc.admin) {
      return res.status(403).json({
        success: false,
        error: "admin_required",
        message: "Admin API key required.",
      });
    }

    req.apiKey = key;
    req.keyMeta = keyDoc;
    next();
  } catch (error) {
    next(error);
  }
};

module.exports = { protect, requireAdmin };
