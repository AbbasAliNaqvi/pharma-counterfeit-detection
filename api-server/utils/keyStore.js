const fs = require("fs");
const path = require("path");

const FILE = path.join(__dirname, "../data/keys.json");

const DEFAULTS = {
  pg_test_key_demo: {
    name: "Demo Key",
    created: new Date().toISOString(),
    requests: 0,
    rateLimit: 100,
    active: true,
    admin: false,
  },
  pg_admin_secret_change_me: {
    name: "Admin Key",
    created: new Date().toISOString(),
    requests: 0,
    rateLimit: 999999,
    active: true,
    admin: true,
  },
};

const load = () => {
  try {
    if (!fs.existsSync(FILE)) {
      fs.mkdirSync(path.dirname(FILE), { recursive: true });
      fs.writeFileSync(FILE, JSON.stringify(DEFAULTS, null, 2));
      return { ...DEFAULTS };
    }
    return JSON.parse(fs.readFileSync(FILE, "utf-8"));
  } catch {
    return { ...DEFAULTS };
  }
};

const save = (keys) => {
  fs.mkdirSync(path.dirname(FILE), { recursive: true });
  fs.writeFileSync(FILE, JSON.stringify(keys, null, 2));
};

module.exports = { load, save };
