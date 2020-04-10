const latentSpace = 300;
const numberSliders = 40;
let pcas = [];
let modelPath = 'famousa-decoder-js/model.json';
let model;
let eigenvalues, eigenvectors, meanData;
let canvas;
let randomButton;
let resetButton;
let legend;

async function LoadModel() {
  console.log('Loading model..');
  model = await tf.loadLayersModel(modelPath);
  console.log('Sucessfully loaded model');

  // create sliders
  for (let i = 0; i < numberSliders; ++i) {
    pca = createSlider(-125, 125, 0, 10);
    // let texts = selectAll('.text');
    pca.input(GenerateFace);
    pca.addClass('slider');
    if (i < 20) {
      let column_sliders = document.getElementsByClassName('slider_column_padded')[0];
      createP(legend.legend[i]).parent(column_sliders);
      pca.parent(column_sliders);

    } else {
      let column_sliders = document.getElementsByClassName('slider_column_padded')[1];
      if (legend.legend[i])
        createP(legend.legend[i]).parent(column_sliders);
      else
        createP("???").parent(column_sliders);
      pca.parent(column_sliders);
    }

    pcas.push(pca);
  }


  // console.log(eigenvalues);

  eigenvectors = Json2Tensor(eigenvectors, 2);
  // console.log(eigenvectors);

  meanData = Json2Tensor(meanData, 1);

  // LatentInterpolation(0, 1, 10);
  GenerateFace();
}

function preload() {
  eigenvectors = loadJSON("eigenvectors.json");
  meanData = loadJSON("meanData.json");
  legend = loadJSON("legend.json");
  LoadModel();
}

function Json2Tensor(x, dim) {
  if (dim == 1) {
    arr = [];
    let n = Object.keys(x).length;
    for (let i = 0; i < n; ++i) {
      arr[i] = x[i];
    }
    return tf.tensor2d(arr, shape = [n, 1]);
  } else if (dim == 2) {
    let n = Object.keys(x).length;
    let m = Object.keys(x[0]).length;
    arr = new Array(n);
    for (let i = 0; i < n; ++i) {
      arr[i] = new Array(m);
      for (let j = 0; j < m; ++j) {
        arr[i][j] = x[i][j];
      }
    }
    return tf.tensor2d(arr, shape = [n, m]);
  }
}


function setup() {
  canvas = createCanvas(256, 256);
  resetButton = createButton("Reset");
  randomButton = createButton("Random Face");
  randomButton.mousePressed(RandomFace);
  resetButton.mousePressed(Reset);
  let img_column = document.getElementById('img_column_padded');
  let random_button = document.getElementById('random_button');
  let reset_button = document.getElementById('reset_button');
  canvas.parent(img_column);
  randomButton.parent(random_button);
  resetButton.parent(reset_button);
  background(0);
  pixelDensity(1);
}

async function GenerateFace() {

  tf.tidy(() => {
    // convert 10, 50 into a vector
    settings = [];
    // let arr = [[pcas[0].value(), pcas[1].value()]];
    for (let i = 0; i < latentSpace; ++i) {

      if (i < numberSliders) {

        // pcas[i].value = random(-100, 100);
        settings[i] = pcas[i].value() / 100.;
        // settings[i] = random(-100, 100) / 100;

      } else
        settings[i] = 0;
    }
    // console.log(settings);

    settings = tf.tensor2d(settings, shape = [300, 1]);

    real_settings = meanData.clone().reshape([1, 300]);

    // let a = tf.mul(eigenvalues, settings);
    let b = tf.mul(eigenvectors, settings);
    let c = tf.sum(b, axis = 0);
    real_settings = real_settings.add(c);

    // real_settings = real_settings.add(settings.dot(eigenvectors.dot(tf.eye(300, 300).mul(eigenvalues))))

    let prediction = model.predict(real_settings).dataSync();
    prediction = tf.tensor3d(prediction, [128, 128, 3]);
    prediction = tf.image.resizeNearestNeighbor(prediction, [256, 256])
    prediction = prediction.as1D().data().then((image) => {
      loadPixels();
      for (let i = 0; i < width; ++i) {
        for (let j = 0; j < height; ++j) {
          let pos = (i + width * j) * 4;
          let index = (i + width * j) * 3;
          // let col = color(prediction[pos], prediction[pos], prediction[pos]);
          // pixels[pos] = red(col);
          // pixels[pos + 1] = green(col);
          // pixels[pos + 2] = blue(col);
          // pixels[pos + 3] = alpha(col);
          pixels[pos] = floor(image[index] * 255);
          pixels[pos + 1] = floor(image[index + 1] * 255);
          pixels[pos + 2] = floor(image[index + 2] * 255);
          pixels[pos + 3] = 255;

        }
      }
      updatePixels()
    });
    // real_settings
    // tf.brow(prediction).print();


    // console.log(prediction)
    // prediction = prediction.as1D();

    // img = loadImage(prediction);
    // let ri_tf = tf.tensor3d(prediction, shape = [64, 64, 3]).mul(255);
    // let t = tf.image.resizeNearestNeighbor(ri_tf, [256, 256])
    // prediction = t.as1D();



  })



}

// function draw() {
//   console.log(tf.memory().numTensors);
// }

function RandomFace() {
  for (let i = 0; i < latentSpace; ++i) {

    if (i < numberSliders) {

      pcas[i].value(random(-50, 50));
      // settings[i] = random(-100, 100) / 100;

    }
  }
  GenerateFace();
  console.log("Random");
}

function Reset() {
  for (let i = 0; i < latentSpace; ++i) {

    if (i < numberSliders) {

      pcas[i].value(0);
      // settings[i] = random(-100, 100) / 100;

    }
  }
  GenerateFace();
  console.log("Reset");
}
