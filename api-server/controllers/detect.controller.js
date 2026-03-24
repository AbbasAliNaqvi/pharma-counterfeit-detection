const { runInference } = require("../utils/inference");

const status = async (req, res, next) => {
  try {
    res.json({
      success: true,
      status: "operational",
      model: "MobileNetV3-Small (ONNX)",
      test_accuracy: 0.96,
      auc_roc: 0.9947,
      f1_score: 0.96,
      classes: ["Authentic", "Counterfeit"],
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    next(err);
  }
};

const analyse = async (req, res, next) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: "no_image",
        message: "No image provided. Form field name must be: image",
      });
    }

    const result = await runInference(req.file.buffer);

    res.json({
      success: true,
      api_key: req.apiKey.slice(0, 8) + "...",
      filename: req.file.originalname,
      size_kb: (req.file.size / 1024).toFixed(1),
      result,
      timestamp: new Date().toISOString(),
    });
  } catch (err) {
    next(err);
  }
};

module.exports = { analyse, status };
