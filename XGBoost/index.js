
async function runExample()

  var x = []; //new float32array(1,11)

  x[0] = document.getElementById('box1').value;
  x[1] = document.getElementById('box2').value;
  x[2] = document.getElementById('box3').value;
  x[3] = document.getElementById('box4').value;
  x[4] = document.getElementById('box5').value;
  x[5] = document.getElementById('box6').value;
  x[6] = document.getElementById('box7').value;
  x[7] = document.getElementById('box8').value;
  x[8] = document.getElementById('box9').value;
  x[9] = document.getElementById('box10').value;
  x[10] = document.getElementById('box11').value;

  let tensorX = new ort.Tensor('float32', x, [1,11]);
  let feeds = {float_input: tensor};
  
  let session = await ort.InferenceSession.create('xgboost_winequality_ort.onnx');

  let result = wait session.run( feeds );
  let outputData = result.variable.data;
  outputData = parseFloat(outputData).toFixed(2);


  //await session.loadModel("./xgb_deployment.onnx");
  //let outputMap = await session.run([tensorX]);
  //let outputData = outputMap.get('output1')
  
  let predictions = document.getElementById('predictions');
//backtick``
predictions.innerHTML=`<hr> Got an output tensor with values:</br>
  <table>
    <tr>
      <td> Quality Rating </td>
      <td id='table0'> ${outputData} </td>
    </tr>
  </table>;

`; //endbacktick``
