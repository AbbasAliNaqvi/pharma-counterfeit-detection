const fs = require("fs");
const path = require("path");

const FILE = path.join(__dirname, "../data/keys.json");
const ADMIN_API_KEY = process.env.ADMIN_API_KEY || "pg_admin_secret";

const buildDefaults = () => ({
  pg_test_key: {
    name: "Demo Key",
    created: new Date().toISOString(),
    requests: 0,
    rateLimit: 100,
    active: true,
    admin: false,
  },
  [ADMIN_API_KEY]: {
    name: "Admin Key",
    created: new Date().toISOString(),
    requests: 0,
    rateLimit: 999999,
    active: true,
    admin: true,
  },
});

const ensureDefaults = (loadedKeys = {}) => {
  const defaults = buildDefaults();
  const nextKeys = { ...loadedKeys };

  nextKeys.pg_test_key = {
    ...defaults.pg_test_key,
    ...(loadedKeys.pg_test_key || {}),
    admin: false,
  };

  Object.keys(nextKeys).forEach((key) => {
    if (nextKeys[key]?.admin) delete nextKeys[key];
  });

  nextKeys[ADMIN_API_KEY] = {
    ...defaults[ADMIN_API_KEY],
    ...(loadedKeys[ADMIN_API_KEY] || {}),
    admin: true,
    active: true,
  };

  return nextKeys;
};

const load = () => {
  try {
    if (!fs.existsSync(FILE)) {
      fs.mkdirSync(path.dirname(FILE), { recursive: true });
      const defaults = buildDefaults();
      fs.writeFileSync(FILE, JSON.stringify(defaults, null, 2));
      return defaults;
    }
    const keys = JSON.parse(fs.readFileSync(FILE, "utf-8"));
    const syncedKeys = ensureDefaults(keys);

    if (JSON.stringify(keys) !== JSON.stringify(syncedKeys)) {
      save(syncedKeys);
    }

    return syncedKeys;
  } catch {
    return buildDefaults();
  }
};

const save = (keys) => {
  fs.mkdirSync(path.dirname(FILE), { recursive: true });
  fs.writeFileSync(FILE, JSON.stringify(keys, null, 2));
};

module.exports = { load, save };
