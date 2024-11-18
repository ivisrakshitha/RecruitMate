const express = require("express");
const mongoose = require("mongoose");
const dotenv = require("dotenv");
const Activity = require("./models/Activity"); // Import the Activity model
const fs = require("fs");
const path = require("path");
const cors = require("cors");
dotenv.config();

const app = express();

app.use(cors());
app.use(express.json());

// Increase payload size limit
app.use(express.json({ limit: "10mb" }));  // Set limit based on expected payload size
app.use(express.urlencoded({ limit: "10mb", extended: true }));

mongoose.connect(process.env.MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
.then(() => console.log("Connected to MongoDB"))
.catch((error) => console.error("MongoDB connection error:", error));

const PORT = process.env.PORT || 3000;

// Create directory for storing screenshots if it doesn't exist
const screenshotDir = path.join(__dirname, "screenshots");
if (!fs.existsSync(screenshotDir)) {
  fs.mkdirSync(screenshotDir);
}

// API endpoint to log activity with a screenshot
app.post("/api/activities", async (req, res) => {
  const { type, timestamp, screenshot } = req.body;
  
  /*if (!screenshot) {
    return res.status(400).send({ error: "Screenshot is required" });
  }*/

  // Convert base64 screenshot data to binary and save it as a file
  /*const screenshotBuffer = Buffer.from(screenshot, "base64");
  const screenshotFilename = `${Date.now()}.png`;
  const screenshotPath = path.join(screenshotDir, screenshotFilename);

  fs.writeFile(screenshotPath, screenshotBuffer, (err) => {
    if (err) {
      console.error("Error saving screenshot:", err);
      return res.status(500).send({ error: "Failed to save screenshot" });
    }
  });
*/
  // Save the activity log to MongoDB
  const activity = new Activity({
    type,
    timestamp,
  //  screenshotPath: `/screenshots/${screenshotFilename}`
  });

  try {
    await activity.save();
    res.status(201).send({ message: "Activity logged successfully" });
  } catch (error) {
    console.error("Error saving activity:", error);
    res.status(500).send({ error: "Failed to log activity" });
  }
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
