import h5py
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.optim as optim

from argparse import ArgumentParser 
from time import perf_counter_ns, perf_counter

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
def model(tu,w,b):
    ''' linear model '''
    return w*tu+b

# tp-> t predicted
def loss_f(tp,tc):
    ''' return the mean of the squared differences of parrallel arrays '''
    squared_diffs=(tp-tc)**2
    return squared_diffs.mean()

@timer
def training_loop(n_epochs,
                  optimizer,
                  params,
                  tutraining,
                  tuvalidate,
                  tctraining,
                  tcvalidate):
    ''' train our model ''' 
    for epoch in range(1, n_epochs+1):
        train_t_p=model(tutraining,*params)
        train_loss=loss_f(train_t_p, tctraining)
        #with torch.nograd():
        with torch.inference_mode():
            val_t_p=model(tuvalidate, *params)
            val_loss=loss_f(val_t_p, tcvalidate)
            assert val_loss.requires_grad==False
        # optimizer is cumulative
        # only for each iteration
        optimizer.zero_grad()
        # back propogate with the optimizer
        train_loss.backward()
        optimizer.step()
        if epoch <= 3 or epoch % 500 == 0:
            print(f'Epoch {epoch}, training loss {train_loss.item():.4f},' ,
                  f'Validation loss {val_loss.item():.4f}')
    return params

def training(n_samp):
    ''' split the data set into training 80~20 '''
    return int(n_samp*0.2)

def shuffle(n_samp):
    return torch.randperm( n_samp )
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
    parser = ArgumentParser(description='Generate Fibonacci Sequence')
    parser.add_argument('--filename', type=str, help='the filename to write to')
    parser.add_argument('--iters', type=int, help='iterations for generating the sequence')
    parser.add_argument('--thresh', type=bool, help='set numpy print to inf')
    args = parser.parse_args()
    file = args.filename
    data = fibonacci( args.iters )
    np_thresh( args.thresh )

    nparr = np.array(data)
    npinfo(nparr)
    #h5write(file, nparr)
    #data = h5read(file)
    txt_write(data, file)

    nparr = np.genfromtxt(file+'.txt', delimiter=",")
    nparr2 = golden_thread(nparr[:])
    #create a bounded threshold to add to the actual values
    #to generate a variance in the synthetic data points
    Y = np.array( [(entry + random.uniform(-0.25,0.25)) for entry in nparr2] )
    file = f'synthetic_data/_{len(Y)}.txt'
    txt_write(Y, file)
    #plot_arr(Y, args.iters)
    X=np.arange(len(Y))
    print(X)
    print(Y)
    #split into test and train data
    #plot dependent and independent variables
    plt.scatter(X,Y)
    plt.title('Synthetic Datapoints')
    plt.savefig(f'plots/golden_thread_scatter_{len(Y)}.png', bbox_inches='tight')
    plt.show()

    """ tt=torch.tensor( nparr )
    n_validate=training(tt.shape[0])
    shuffled_indices=shuffle(tt.shape[0])
    # using the size of 20% of the data for slicing
    training_indices=shuffled_indices[:-n_validate]
    validation_indices=shuffled_indices[-n_validate:]
    # instantiate training tensors
    tutraining_t=tu_t[training_indices]
    tctraining_t=tc_t[training_indices]
    # instantiate validation tensors
    tuvalidate_t=tu_t[validation_indices]
    tcvalidate_t=tc_t[validation_indices]
    # normalize the training data
    tutraining_normt=0.1*tutraining_t
    tuvalidate_normt=0.1*tuvalidate_t

    '''forming a computational graph'''
    #call everything
    params = torch.tensor( [1.0,0.0], requires_grad=True)
    # delta-step defines how much weights and biases get modified
    learning_rate= .01

    optimizer=optim.SGD( [params], lr=learning_rate)
    result=training_loop( n_epochs=1000,
                        optimizer=optimizer,
                        params = params,
                        tutraining = tutraining_normt,
                        tuvalidate = tuvalidate_normt, 
                        tctraining = tctraining_t,
                        tcvalidate = tcvalidate_t
                        )

    print(result)
    print(type(result))
    t_p = model(tuvalidate_normt, *params) """


if __name__ == "__main__":
    main()
