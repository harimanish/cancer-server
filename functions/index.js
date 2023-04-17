const express = require('express');
const tf = require('@tensorflow/tfjs-node');
const axios = require('axios');
const path = require('path');
const functions = require('firebase-functions');


const app = express();
app.use(express.json());

const port = 3000;
const MODEL_PATH = {
  skin: path.join(__dirname, 'skin-model', 'skin-model.json'),
  invasive: path.join(__dirname, 'invasive-model', 'invasive-model.json'),
  meta: path.join(__dirname, 'meta-model', 'meta-model.json'),
};
const CLASSES = {
  skin: {
    0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
    1: 'Basal Cell Carcinoma',
    2: 'Benign Keratosis',
    3: 'Dermatofibroma',
    4: 'Melanoma',
    5: 'Melanocytic Nevi',
    6: 'Vascular skin lesion',
  },
  invasive: {
    0: 'Normal',
    1: 'Invasive Ductal Carcinoma'
  },
  meta: {
    0: 'Normal',
    1: 'Metastatic Tissue',
  }
};
const scalarOffset = {
  skin: 127.5,
  invasive: 255.0,
  meta: 255.0
}
const imageSize = {
  skin: [224, 224],
  invasive: [50, 50],
  meta: [96, 96],
};

const preprocessImage = (imageBuffer, modelType) => {
  const imageTensor = tf.node.decodeImage(imageBuffer);
  const size = imageSize[modelType];
  const resized = tf.image.resizeNearestNeighbor(imageTensor, size);
  const casted = resized.cast('float32');
  const expanded = casted.expandDims();
  const offset = scalarOffset[modelType];
  return expanded.sub(offset).div(offset);
};

const predict = async (imageUrl, modelType) => {
  try {
    const response = await axios.get(imageUrl, { responseType: 'arraybuffer' });
    const tensor = preprocessImage(response.data, modelType);
    const model = await tf.loadLayersModel(`file://${MODEL_PATH[modelType]}`);
    const predictions = await model.predict(tensor).data();
    const classes = CLASSES[modelType];
    const topPredictions = Array.from(predictions)
      .map((probability, i) => ({ probability, className: classes[i] }))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, modelType === 'skin' ? 5 : 2);

    const fixed = modelType === "skin" ? 3 : 6

    topPredictions.forEach(function (p) {
      `${p.className}: ${p.probability.toFixed(fixed)}`;
    })

    return topPredictions;
  }
  catch (error) {
    console.error(error);
    throw new Error(`Error predicting image: ${error.message}`);
  }
};

app.post('/predict/:modelType', async (req, res) => {
  const { modelType } = req.params;

  if (!['skin', 'invasive', 'meta'].includes(modelType)) {
    return res.status(400).send('Invalid model type');
  }
  const { url } = req.body;
  if (!url) {
    return res.status(400).send('Missing image URL');
  }
  try {
    const topPredictions = await predict(url, modelType);
    res.json(topPredictions);
  } catch (err) {
    console.error(err);
    res.status(500).send('Internal Server Error');
  }
});

// app.listen(port, () => {
//     console.log(`Server started on port ${port}`);
// });
exports.app = functions.https.onRequest(app);