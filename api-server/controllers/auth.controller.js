const { v4: uuidv4 } = require("uuid");
const KeyStore = require("../utils/keyStore");

const register = async (req, res, next) => {
  try {
    const { name, email, use } = req.body;

    if (!name || !email || !use) {
      return res
        .status(400)
        .json({ success: false, message: "name, email and use are required." });
    }

    const existing = await KeyStore.findByEmail(email);
    if (existing) {
      return res
        .status(409)
        .json({
          success: false,
          message: "A key already exists for this email.",
        });
    }

    const key = "pg_" + uuidv4().replace(/-/g, "").slice(0, 24);

    await KeyStore.createKey({
      key,
      name,
      email,
      use,
      requests: 0,
      rateLimit: 100,
      active: true,
      admin: false,
    });

    res.status(201).json({
      success: true,
      api_key: key,
      name,
      rate_limit: 100,
      message: "Store this key securely.",
    });
  } catch (error) {
    next(error);
  }
};

const createKey = async (req, res, next) => {
  try {
    const { name = "New Key", email = null, rateLimit = 1000 } = req.body;
    const key = "pg_" + uuidv4().replace(/-/g, "").slice(0, 24);

    await KeyStore.createKey({
      key,
      name,
      email,
      requests: 0,
      rateLimit: parseInt(rateLimit, 10),
      active: true,
      admin: false,
    });

    res.status(201).json({
      success: true,
      api_key: key,
      name,
      rate_limit: rateLimit,
      message: "Store this key securely.",
    });
  } catch (error) {
    next(error);
  }
};

const revokeKey = async (req, res, next) => {
  try {
    const match = await KeyStore.revokeByPrefix(req.params.prefix);
    if (!match) {
      return res.status(404).json({ success: false, error: "key_not_found" });
    }

    res.json({
      success: true,
      message: `Key ${req.params.prefix}... revoked.`,
    });
  } catch (error) {
    next(error);
  }
};

module.exports = { register, createKey, revokeKey };
