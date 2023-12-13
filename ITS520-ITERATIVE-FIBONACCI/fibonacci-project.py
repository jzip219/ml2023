import h5py
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.optim as optim
from torch.autograd import Variable

from argparse import ArgumentParser 
from datetime import datetime
from time import perf_counter_ns, perf_counter
from lregnet import LinearRegressionModel

import onnx
import onnxruntime as ort
import torch.onnx

def timer(func):
    def wrapper(*args, **kwargs):
        start = perf_counter_ns()
        result = func(*args, **kwargs)
        end = perf_counter_ns() - start
        print(f'"{func.__name__}(s)": {end/1000000000}')
        return result
    return wrapper

def np_thresh(thresh=False):
    if thresh:
        np.set_printoptions(threshold=np.inf)

#########################################################
@timer
def training_loop(model,x_train,y_train,optimizer,criterion,epochs):
    for epoch in range(epochs):
        inputs = x_train
        labels = y_train
        # Clear gradient buffers because we don't want any 
        # gradient from previous epoch to carry forward, 
        # do not want to accumulate gradients
        optimizer.zero_grad()
        # get output from the model, given the inputs
        outputs = model(inputs)
        # get loss for the predicted output
        loss = criterion(outputs, labels)
        #print(loss)
        # get gradients w.r.t to parameters
        loss.backward()
        # update parameters
        optimizer.step()
        #print('epoch {}, loss {}'.format(epoch, loss.item()))
    return model
#########################################################
@timer
def fibonacci(n):
    #fib_seq = np.arange(2, dtype=int)
    fib_seq = [0,1]
    for idx in range(1,n):
        #np.append(fib_seq, fib_seq[idx]+fib_seq[idx-1])
        fib_seq.append(fib_seq[idx]+fib_seq[idx-1])
    return fib_seq

@timer
def golden_thread(data):
    phi_approx = [0]
    for idx in range(2, len(data)):
        phi_approx.append(data[idx]/data[idx-1])
        #print(f'phi_approx[{idx}] = {phi_approx[idx-1]} ')
    return phi_approx  

def txt_write(data, filename):
    with open(filename+'.txt', 'w') as fobj:
        fobj.write(str(data)[1:-1])

def npinfo(nparr):
    print(f'nparr:\n{type(nparr)}')
    print(f'shape:\n{nparr.shape}')
    print(f'size:\n{nparr.size}')
    #print(nparr)

def plot_arr(data, entries, ts):
    X = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(10, 5))
    #create a line where it equals 1.618
    ax.axhline(y=(math.sqrt(5)+1)/2, color='r', linestyle='-')
    ax.set_title('Golden Thread')
    ax.set_xlabel('Fibonacci Sequence')
    ax.set_ylabel('Current/Previous')
    ax.spines[['right','top']].set_visible(False)
    plt.plot(X, data)
    plt.savefig(f'plots/{ts}_{entries}.png', bbox_inches='tight')
    plt.show()

def convert_to_onnx(model, input_size, onnx_file_name):
    dummy_input = torch.randn(input_size)
    torch.onnx.export(model, dummy_input, onnx_file_name, export_params=True)

""" def onnx_predict(onnx_model, x_test):
    ort_session = ort.InferenceSession(onnx_model)
    ort_inputs = {ort_session.get_inputs()[0].name: x_test.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0] """

""" def onnx_predict(onnx_model, x_test):
    ort_session = ort.InferenceSession(onnx_model)
    # Reshape x_test to match the expected input shape of the model
    # Assuming the model expects a single feature input
    x_test_reshaped = x_test.reshape(-1, 1).numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: x_test_reshaped}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs[0] """

def main():
    ###################################################################
    parser = ArgumentParser(description='Generate Fibonacci Sequence')
    parser.add_argument('--filename', type=str, help='the filename to write to')
    parser.add_argument('--iters', type=int, help='iterations for generating the sequence')
    parser.add_argument('--epochs', type=int, help='number of iters for model training')
    parser.add_argument('--learn_rate', type=float, help='learning rate for model training')
    parser.add_argument('--thresh', type=bool, help='set numpy print to inf')
    args = parser.parse_args()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    file = 'fibonacci_sequence/'+ts+args.filename
    data = fibonacci( args.iters )
    epochs = args.epochs
    learn_rate = args.learn_rate
    np_thresh( args.thresh )
    ###################################################################
    nparr = np.array(data)
    npinfo(nparr)
    #txt_write(data, file)
    ###################################################################
    #nparr = np.genfromtxt(file+'.txt', delimiter=",")
    nparr2 = golden_thread(nparr[:])
    #create a bounded threshold to add to the actual values
    #to generate a variance in the synthetic data points
    Y = np.array( [(entry + random.uniform(-0.25,0.25)) for entry in nparr2] )
    ## keep outliers?
    #Y=nparr2
    ## get rid of outliers?
    Y = Y[3:]
    file = 'synthetic_data/'+ts+'_{}'.format(len(Y))
    #txt_write(Y, file)
    plot_arr(Y, args.iters, ts)
    X=np.arange(len(Y))
    ###################################################################
    # Convert data to a tensor
    data_tensor = torch.tensor(Y, dtype=torch.float32)
    # Reshape data to fit the (x, y) format for regression
    x = torch.arange(len(data_tensor)).unsqueeze(1).float()  # Independent variable (index)
    y = data_tensor.unsqueeze(1)  # Dependent variable (value)
    ###################################################################
    # Split data into training and testing sets
    train_size = int(0.8 * len(data_tensor))
    test_size = len(data_tensor) - train_size
    x_train, x_test = torch.split(x, [train_size, test_size], dim=0)
    y_train, y_test = torch.split(y, [train_size, test_size], dim=0)
    ###################################################################
    # Create the model
    model = LinearRegressionModel()
    # Loss and Optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
    ###################################################################
    model = training_loop(model,x_train,y_train,optimizer,criterion,epochs)
    ###################################################################
    with torch.no_grad(): # we don't need gradients in the testing phase
        #predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
        predicted = model(x_train).data.numpy()
    #print(predicted)
    ###################################################################
    plt.clf()
    plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
    plt.plot(x_train, predicted, '--', label='Predictions',)
    plt.legend(loc='best')
    plt.savefig(f'plots/{ts}.png', bbox_inches='tight')
    plt.show()
    ###################################################################

    """ # Convert to ONNX after training
    onnx_model_name = "linear_regression_model.onnx"
    convert_to_onnx(model, (1, 1), onnx_model_name)
    # Use ONNX Runtime for inference
    with torch.no_grad():
        predicted = onnx_predict(onnx_model_name, x_train) """

    ###################################################################
    # Extracting the model parameters for plotting
    [w, b] = list(model.to('cpu').parameters())
    # Detach the parameters from the computation graph 
    # and convert to NumPy for plotting
    weight = w.detach().numpy()[0][0]  # Assuming model has single weight
    bias = b.detach().numpy()[0]       # Bias term
    #print weight and bias
    print(f'datapoints: {args.iters}')   
    print(f'learn_rate: {learn_rate}')
    print(f'epochs: {epochs}')
    print(f'weight: {weight}')
    print(f'bias: {bias}')
    print(f'y = w * x + b')
    x=int(input('enter an integer: '))
    print(f'y = {weight}*{x} + {bias}')
    print(f'y = {weight * x + bias}')
    ###################################################################
    # Collect the data in a dictionary
    data = {
        "datapoints": args.iters,
        "learn_rate": learn_rate,
        "epochs": epochs,
        "weight": float(weight),
        "bias": float(bias),
        "equation": "y = w * x + b",
    }
    # Input an integer
    # Calculate y
    y = weight * x + bias
    # Add y to the data dictionary
    data["x"] = x
    # Add y to the data dictionary
    data["y"] = float(y)
    # Convert the data dictionary to a JSON string
    json_str = json.dumps(data, indent=4)
    # Save the JSON string to a file
    with open("weight_and_bias/"+ts+".json", "w") as json_file:
        json_file.write(json_str)
    # Print the JSON string
    print(json_str)
    ###################################################################









    """     # Training loop
    for epoch in range(1000):  # Number of iterations
        # Forward pass
        pred_y = model(x)
        loss = criterion(pred_y, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item()}')
    
    
    
    # Extracting the model parameters for plotting
    [w, b] = list(model.parameters())
    # Detach the parameters from the computation graph and convert to NumPy for plotting
    weight = w.detach().numpy()[0][0]  # Assuming model has single weight
    bias = b.detach().numpy()[0]       # Bias term
    #print weight and bias
    print(f'weight: {weight}')
    print(f'bias: {bias}')
    print(w.detach().numpy()[0][0])



    # Now, weight and bias can be used to create the line of best fit
    # For example, you can use them to calculate the predicted y values
    #predicted_y = weight * x.numpy() + bias

    # Predicted values for plotting
    predicted_y = model(x).detach().numpy()
    print(f'pred y:\n{predicted_y}')
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x.numpy(), y.numpy(), label='Original Data')
    plt.plot(x.numpy(), predicted_y, color='red', label='Fitted Line')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Linear Regression Fit')
    plt.legend()
    plt.grid(True)
    plt.show() """




if __name__ == "__main__":
    main()
