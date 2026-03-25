const path = require("path");
require("dotenv").config({ path: path.join(__dirname, ".env") });
const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const morgan = require("morgan");

const detectRoutes = require("./routes/detect.routes");
const authRoutes = require("./routes/auth.routes");
const adminRoutes = require("./routes/admin.routes");
const { notFound, errorHandler } = require("./middleware/error.middleware");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors());
app.use(express.json());
app.use(morgan("dev"));

app.use(express.static(path.join(__dirname, "public")));

app.use("/v1/detect", detectRoutes);
app.use("/v1/auth", authRoutes);
app.use("/v1/admin", adminRoutes);

app.get("/v1", (req, res) => {
  res.json({
    service: "PharmaGuard API",
    version: "1.0.0",
    status: "operational",
    endpoints: {
      "POST /v1/detect/analyse": "Classify medicine image",
      "POST /v1/detect/analyse/gradcam": "Classify + Grad-CAM heatmap",
      "GET  /v1/detect/status": "Model health check",
      "POST /v1/auth/register": "Get free API key (public)",
      "POST /v1/auth/keys": "Generate API key (admin)",
      "GET  /v1/admin/keys": "List all keys (admin)",
      "GET  /v1/admin/stats": "Usage stats (admin)",
    },
  });
});

app.use(notFound);
app.use(errorHandler);

app.listen(PORT, () => {
  console.log(`\nPharmaGuard API is Active`);
});

module.exports = app;
