const KeyStore = require("../utils/keyStore");

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

module.exports = { getAllKeys, getStats };
