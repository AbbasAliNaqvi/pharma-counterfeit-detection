const mongoose = require("mongoose");

const apiKeySchema = new mongoose.Schema(
  {
    key: {
      type: String,
      required: true,
      unique: true,
      index: true,
      trim: true,
    },
    name: {
      type: String,
      required: true,
      trim: true,
    },
    email: {
      type: String,
      default: null,
      trim: true,
    },
    use: {
      type: String,
      default: null,
      trim: true,
    },
    requests: {
      type: Number,
      default: 0,
      min: 0,
    },
    rateLimit: {
      type: Number,
      default: 100,
      min: 1,
    },
    active: {
      type: Boolean,
      default: true,
    },
    admin: {
      type: Boolean,
      default: false,
    },
  },
  { timestamps: true },
);

module.exports = mongoose.models.ApiKey || mongoose.model("ApiKey", apiKeySchema);
