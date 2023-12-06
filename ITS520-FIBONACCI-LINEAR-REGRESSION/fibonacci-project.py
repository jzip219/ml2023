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
from time import perf_counter_ns, perf_counter
from lregnetpy import LinearRegressionModel

def timer(func):
    def wrapper(*args, **kwargs):
        start = perf_counter_ns()
        result = func(*args, **kwargs)
        end = perf_counter_ns() - start
        print(f'{func.__name__} took {end} ns')
        return result
    return wrapper

def np_thresh(thresh=False):
    if thresh:
        np.set_printoptions(threshold=np.inf)

#########################################################
@timer
def training_loop(model,x_train,y_train,optimizer,criterion,epochs):
    for epoch in range(epochs):
        inputs = x_train#Variable(torch.from_numpy(x_train))
        labels = y_train#Variable(torch.from_numpy(y_train))

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
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
def h5write(filename, data):
    with h5py.File(filename+'.h5', 'w') as hf: 
        hf.create_dataset("fibonacci",  data=data ) 

@timer
def h5read(filename):
    with h5py.File(filename+'.h5', 'r') as hf:
        data = hf['fibonacci'][:]
    return data

@timer
def fibonacci(n):
    #fib_seq = np.arange(2, dtype=int)
    fib_seq = [0,1]
    for idx in range(1,n):
        #np.append(fib_seq, fib_seq[idx]+fib_seq[idx-1])
        fib_seq.append(fib_seq[idx]+fib_seq[idx-1])
    return fib_seq

def json_dump(data, filename):
    dflags = str(data.flags)
    data_str = '{"data:": '
    data_str += str({"data shape": str(data.shape),
                 "data type": str(data.dtype),
                "data size": str(data.size),
                "data ndim": str(data.ndim),
                "data nbytes": str(data.nbytes),
                "data itemsize": str(data.itemsize),
                "data strides": str(data.strides),
                #"data flags": dflags #np
                }) +'}'
    json_string = json.dumps(data_str)
    return json_string

def json_logs(json_string, file):
    with open(file+'.json', 'w') as fobj:
        json.dump(json_string, fobj, indent=4)

def json_read(file):
    with open(file+'.json', 'r') as fobj:
        json_data = json.load(fobj)
    print(json_data)

def txt_write(data, filename):
    with open(filename+'.txt', 'w') as fobj:
        fobj.write(str(data)[1:-1])

def npinfo(nparr):
    print(f'nparr:\n{type(nparr)}')
    print(f'shape:\n{nparr.shape}')
    print(f'size:\n{nparr.size}')
    print(nparr)

@timer
def golden_thread(data):
    phi_approx = [0]
    for idx in range(2, len(data)):
        phi_approx.append(data[idx]/data[idx-1])
        #print(f'phi_approx[{idx}] = {phi_approx[idx-1]} ')
    return phi_approx        

def plot_arr(data, entries):
    X = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(10, 5))
    #create a line where it equals 1.618
    ax.axhline(y=(math.sqrt(5)+1)/2, color='r', linestyle='-')
    ax.set_title('Golden Thread')
    ax.set_xlabel('Fibonacci Sequence')
    ax.set_ylabel('Current/Previous')
    ax.spines[['right','top']].set_visible(False)
    plt.plot(X, data)
    plt.savefig(f'plots/golden_thread_{entries}.png', bbox_inches='tight')
    plt.show()
    
def main():
    ###################################################################
    parser = ArgumentParser(description='Generate Fibonacci Sequence')
    parser.add_argument('--filename', type=str, help='the filename to write to')
    parser.add_argument('--iters', type=int, help='iterations for generating the sequence')
    parser.add_argument('--thresh', type=bool, help='set numpy print to inf')
    args = parser.parse_args()
    file = args.filename
    data = fibonacci( args.iters )
    np_thresh( args.thresh )
    ###################################################################

    nparr = np.array(data)
    npinfo(nparr)
    #h5write(file, nparr)
    #data = h5read(file)
    txt_write(data, file)
    ###################################################################

    nparr = np.genfromtxt(file+'.txt', delimiter=",")
    nparr2 = golden_thread(nparr[:])
    #create a bounded threshold to add to the actual values
    #to generate a variance in the synthetic data points
    Y = np.array( [(entry + random.uniform(-0.25,0.25)) for entry in nparr2] )
    file = f'synthetic_data/_{len(Y)}'
    txt_write(Y, file)
    #plot_arr(Y, args.iters)
    X=np.arange(len(Y))
    #print(X)
    #print(Y)
    #split into test and train data
    #plot dependent and independent variables
    """ fig, ax = plt.subplots(figsize=(10, 5))
    plt.scatter(X,Y)
    plt.title('Synthetic Datapoints')
    plt.savefig(f'plots/golden_thread_scatter_{len(Y)}.png', bbox_inches='tight')
    plt.show() """
    ###################################################################

    # Convert data to a tensor
    data_tensor = torch.tensor(Y, dtype=torch.float32)
    # Normalize the data (Optional but recommended)
    #data_tensor = (data_tensor - data_tensor.mean()) / data_tensor.std()

    
    # Reshape data to fit the (x, y) format for regression
    x = torch.arange(len(data_tensor)).unsqueeze(1).float()  # Independent variable (index)
    y = data_tensor.unsqueeze(1)  # Dependent variable (value)
    """ # Normalize x
    x = (x - x.mean()) / x.std() """

    # Split data into training and testing sets
    train_size = int(0.8 * len(data_tensor))
    test_size = len(data_tensor) - train_size
    x_train, x_test = torch.split(x, [train_size, test_size], dim=0)
    y_train, y_test = torch.split(y, [train_size, test_size], dim=0)
    ###################################################################

    # Create the model
    model = LinearRegressionModel()
    epochs = 100000
    learn_rate = 1/10000
    # Loss and Optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

    model = training_loop(model,x_train,y_train,optimizer,criterion,epochs)
    ###################################################################
    with torch.no_grad(): # we don't need gradients in the testing phase
        #predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
        predicted = model(x_train).data.numpy()
    #print(predicted)

    plt.clf()
    plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)
    plt.plot(x_train, predicted, '--', label='Predictions',)
    plt.legend(loc='best')
    plt.savefig(f'plots/golden_thread_{len(Y)}_regression.png', bbox_inches='tight')
    plt.show()

    # Extracting the model parameters for plotting
    [w, b] = list(model.parameters())
    # Detach the parameters from the computation graph and convert to NumPy for plotting
    weight = w.detach().numpy()[0][0]  # Assuming model has single weight
    bias = b.detach().numpy()[0]       # Bias term
    #print weight and bias
    print(f'weight: {weight}')
    print(f'bias: {bias}')










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
