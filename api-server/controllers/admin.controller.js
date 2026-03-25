const KeyStore = require("../utils/keyStore");

const login = (req, res) => {
  const { id, password } = req.body;

  if (!id || !password) {
    return res.status(400).json({
      success: false,
      message: "Admin ID and password are required.",
    });
  }

  const adminId = process.env.ADMIN_ID;
  const adminPassword = process.env.ADMIN_PASSWORD;
  const adminApiKey = process.env.ADMIN_API_KEY;

  if (!adminId || !adminPassword || !adminApiKey) {
    return res.status(500).json({
      success: false,
      message: "Admin credentials are not configured.",
    });
  }

  if (id !== adminId || password !== adminPassword) {
    return res.status(401).json({
      success: false,
      message: "Invalid credentials.",
    });
  }

  res.json({
    success: true,
    api_key: adminApiKey,
  });
};

const getAllKeys = (req, res) => {
  const keys = KeyStore.load();
  res.json({
    success: true,
    total_keys: Object.keys(keys).length,
    keys: Object.entries(keys).map(([k, v]) => ({
      prefix: k.slice(0, 8) + "...",
      name: v.name,
      requests: v.requests,
      rate_limit: v.rateLimit,
      active: v.active,
      admin: v.admin,
      created: v.created,
    })),
  });
};

const getStats = (req, res) => {
  const keys = KeyStore.load();
  res.json({
    success: true,
    total_keys: Object.keys(keys).length,
    active_keys: Object.values(keys).filter((k) => k.active).length,
    total_requests: Object.values(keys).reduce((s, k) => s + k.requests, 0),
  });
};

module.exports = { login, getAllKeys, getStats };
