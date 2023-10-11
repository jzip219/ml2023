
async function float32array()

  var x = new float32array(1,11)

  x[0] = document.getElementById('box1

let tensorX = new onnx.Tensor(x, float32, [1,11]);

let session = new onnx.InferenceSession();
await session.loadModel("./XGB_Winedata.py")
