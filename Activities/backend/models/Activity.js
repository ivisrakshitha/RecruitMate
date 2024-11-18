const mongoose = require("mongoose");

const activitySchema = new mongoose.Schema({
  type: { type: String, required: true },
  timestamp: { type: Date, default: Date.now },
  screenshotPath: { type: String },
});

const Activity = mongoose.model("Activity", activitySchema);

module.exports = Activity;
