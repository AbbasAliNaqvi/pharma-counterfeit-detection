const notFound = (req, res, next) => {
  res.status(404).json({
    success: false,
    error: "not_found",
    message: `${req.method} ${req.originalUrl} not found.`,
  });
};

const errorHandler = (err, req, res, next) => {
  console.error("[ERROR]", err.message);
  const status = res.statusCode !== 200 ? res.statusCode : 500;
  res.status(status).json({
    success: false,
    error: "server_error",
    message: err.message || "Unexpected error.",
  });
};

module.exports = { notFound, errorHandler };
