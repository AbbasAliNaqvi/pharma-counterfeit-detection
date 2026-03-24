const KeyStore = require("../utils/keyStore");

const log = {};

const rateLimit = (req, res, next) => {
  const key = req.apiKey;
  const limit = req.keyMeta.rateLimit || 100;
  const now = Date.now();
  const day = 86400000;

  if (!log[key]) log[key] = [];
  log[key] = log[key].filter((t) => now - t < day);

  if (log[key].length >= limit) {
    return res.status(429).json({
      success: false,
      error: "rate_limit_exceeded",
      message: `Daily limit of ${limit} requests reached.`,
      resets_at: new Date(log[key][0] + day).toISOString(),
    });
  }

  log[key].push(now);

  const keys = KeyStore.load();
  if (keys[key]) {
    keys[key].requests += 1;
    KeyStore.save(keys);
  }

  res.set("X-RateLimit-Limit", limit);
  res.set("X-RateLimit-Remaining", limit - log[key].length);
  next();
};

module.exports = { rateLimit };
