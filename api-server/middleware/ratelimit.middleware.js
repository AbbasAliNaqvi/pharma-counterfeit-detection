const KeyStore = require("../utils/keyStore");

const log = {};

const rateLimit = async (req, res, next) => {
  try {
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

    await KeyStore.incrementRequests(key);

    res.set("X-RateLimit-Limit", limit);
    res.set("X-RateLimit-Remaining", limit - log[key].length);
    next();
  } catch (error) {
    next(error);
  }
};

module.exports = { rateLimit };
