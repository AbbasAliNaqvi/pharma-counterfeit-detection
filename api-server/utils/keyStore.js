const ApiKey = require("../models/ApiKey");

const ADMIN_API_KEY = process.env.ADMIN_API_KEY || "pg_admin_secret";

const ensureDefaults = async () => {
  await ApiKey.findOneAndUpdate(
    { key: "pg_test_key" },
    {
      $setOnInsert: {
        key: "pg_test_key",
        name: "Demo Key",
        requests: 0,
        rateLimit: 100,
        active: true,
        admin: false,
      },
    },
    { upsert: true, new: true },
  );

  await ApiKey.deleteMany({
    admin: true,
    key: { $ne: ADMIN_API_KEY },
  });

  await ApiKey.findOneAndUpdate(
    { key: ADMIN_API_KEY },
    {
      $set: {
        name: "Admin Key",
        rateLimit: 999999,
        active: true,
        admin: true,
      },
      $setOnInsert: {
        key: ADMIN_API_KEY,
        requests: 0,
      },
    },
    { upsert: true, new: true },
  );
};

const findByKey = async (key) => {
  return ApiKey.findOne({ key }).lean();
};

const findByEmail = async (email) => {
  return ApiKey.findOne({ email }).lean();
};

const createKey = async (payload) => {
  const doc = await ApiKey.create(payload);
  return doc.toObject();
};

const revokeByPrefix = async (prefix) => {
  const doc = await ApiKey.findOne({
    key: { $regex: `^${prefix}` },
  });

  if (!doc) return null;

  doc.active = false;
  await doc.save();
  return doc.toObject();
};

const listKeys = async () => {
  return ApiKey.find({}).sort({ createdAt: -1 }).lean();
};

const getStats = async () => {
  const [totalKeys, activeKeys, requestAgg] = await Promise.all([
    ApiKey.countDocuments(),
    ApiKey.countDocuments({ active: true }),
    ApiKey.aggregate([
      {
        $group: {
          _id: null,
          totalRequests: { $sum: "$requests" },
        },
      },
    ]),
  ]);

  return {
    total_keys: totalKeys,
    active_keys: activeKeys,
    total_requests: requestAgg[0]?.totalRequests || 0,
  };
};

const incrementRequests = async (key) => {
  await ApiKey.updateOne({ key }, { $inc: { requests: 1 } });
};

module.exports = {
  ensureDefaults,
  findByKey,
  findByEmail,
  createKey,
  revokeByPrefix,
  listKeys,
  getStats,
  incrementRequests,
};
