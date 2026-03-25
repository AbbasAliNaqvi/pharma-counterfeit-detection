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

const getAllKeys = async (req, res, next) => {
  try {
    const keys = await KeyStore.listKeys();
    res.json({
      success: true,
      total_keys: keys.length,
      keys: keys.map((keyDoc) => ({
        prefix: keyDoc.key.slice(0, 8) + "...",
        name: keyDoc.name,
        email: keyDoc.email,
        requests: keyDoc.requests,
        rate_limit: keyDoc.rateLimit,
        active: keyDoc.active,
        admin: keyDoc.admin,
        created: keyDoc.createdAt,
      })),
    });
  } catch (error) {
    next(error);
  }
};

const getStats = async (req, res, next) => {
  try {
    const stats = await KeyStore.getStats();
    res.json({ success: true, ...stats });
  } catch (error) {
    next(error);
  }
};

module.exports = { login, getAllKeys, getStats };
