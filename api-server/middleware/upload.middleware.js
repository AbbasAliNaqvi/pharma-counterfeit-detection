const multer = require("multer");

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 20 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const allowed = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
    allowed.includes(file.mimetype)
      ? cb(null, true)
      : cb(new Error("Only JPEG, PNG, and WEBP images accepted."), false);
  },
}).single("image");

module.exports = (req, res, next) => {
  upload(req, res, (err) => {
    if (err instanceof multer.MulterError) {
      return res
        .status(400)
        .json({ success: false, error: "upload_error", message: err.message });
    }
    if (err) {
      return res
        .status(422)
        .json({ success: false, error: "invalid_file", message: err.message });
    }
    next();
  });
};
