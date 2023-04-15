const express = require("express");
const tf = require("@tensorflow/tfjs-node");
const multer = require("multer");
const fs = require("fs");
const path = require('path');
const functions = require("firebase-functions");
const axios = require("axios");

const app = express();
const port = 5000;

app.use(express.json());
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  next();
});

const TARGET_CLASSES = {
  0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
  1: 'Basal Cell Carcinoma',
  2: 'Benign Keratosis',
  3: 'Dermatofibroma',
  4: 'Melanoma',
  5: 'Melanocytic Nevi',
  6: 'Vascular skin lesion'
};

const loadModel = async () => {
  const modelPath = path.join(__dirname, 'model', 'model.json');
  const model = await tf.loadLayersModel(`file://${modelPath}`);
  return model;
};

const preprocessImage = (imageBuffer) => {
  const tensor = tf.node.decodeImage(imageBuffer);
  const resized = tf.image.resizeNearestNeighbor(tensor, [224, 224]);
  const casted = resized.cast("float32");
  const expanded = casted.expandDims();
  const offset = tf.scalar(127.5);
  return expanded.sub(offset).div(offset);
};

app.post('/predict', async (req, res) => {
  try {
    const imageUrl = req.body.url;
    const response = await axios.get(imageUrl, { responseType: "arraybuffer" });
    const tensor = preprocessImage(response.data);
    const model = await loadModel();
    const predictions = await model.predict(tensor).data();
    const top5 = Array.from(predictions)
      .map((p, i) => ({
        probability: p,
        className: TARGET_CLASSES[i],
      }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 5);
    console.log(top5);
    res.json(top5);
  } catch (err) {
    console.error(err);
    res.status(500).send('Internal Server Error');
  }
});

app.listen(port, () => {
  console.log(`Server started on port ${port}`);
});
exports.app = functions.https.onRequest(app);
