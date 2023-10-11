
async function float32array()

  var x = new float32array(1,11)

  x[0] = document.getElementById('box1').value;
  x[1] = document.getElementById('box1').value;
  x[2] = document.getElementById('box1').value;
  x[3] = document.getElementById('box1').value;
  x[4] = document.getElementById('box1').value;
  x[5] = document.getElementById('box1').value;
  x[6] = document.getElementById('box1').value;
  x[7] = document.getElementById('box1').value;
  x[8] = document.getElementById('box1').value;
  x[9] = document.getElementById('box1').value;
  x[10] = document.getElementById('box1').value;

let tensorX = new onnx.Tensor(x, float32, [1,11]);

let session = new onnx.InferenceSession();
await session.loadModel("./xgb_deployment.onnx");
let outputMap = await session.run([)
