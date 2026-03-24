const { v4: uuidv4 } = require("uuid");
const KeyStore = require("../utils/keyStore");

const createKey = (req, res) => {
  const { name = "New Key", rateLimit = 1000 } = req.body;
  const key = "pg_" + uuidv4().replace(/-/g, "").slice(0, 24);
  const keys = KeyStore.load();

  keys[key] = {
    name,
    created: new Date().toISOString(),
    requests: 0,
    rateLimit: parseInt(rateLimit),
    active: true,
    admin: false,
  };

  KeyStore.save(keys);

  res.status(201).json({
    success: true,
    api_key: key,
    name,
    rate_limit: rateLimit,
    message: "Store this key securely.",
  });
};

const revokeKey = (req, res) => {
  const keys = KeyStore.load();
  const match = Object.keys(keys).find((k) => k.startsWith(req.params.prefix));

  if (!match) {
    return res.status(404).json({ success: false, error: "key_not_found" });
  }

  keys[match].active = false;
  KeyStore.save(keys);
  res.json({ success: true, message: `Key ${req.params.prefix}... revoked.` });
};

module.exports = { createKey, revokeKey };
