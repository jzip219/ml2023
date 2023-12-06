import numpy as np
import torch as torch
from matplotlib import pyplot as plt
import torch.optim as optim
import imageio
import os, time
#https://numpy.org/doc/stable/reference/

def gen_fareneheit():
    ''' generate a numpy array of a specified range 
    of random simulated temperature readings in farenheit '''
    MIN_VAL=-33.0
    MAX_VAL=114.0
    ROWS=13000
    COLS=1
    return np.random.uniform(MIN_VAL, MAX_VAL, size=(ROWS,COLS))

def F2C():
    ''' convert farenheit values to celsius 
    C = (F-32) * 5/9 '''
    global F
    F=np.append(F,((F-32)*5/9),axis=1)

def csv(F):
    ''' save a comma delimtted numpy array formatted to
    three decimal places to a .csv file extension '''
    np.savetxt('degrees.csv', F, fmt='%.3f', delimiter=",")

def nparray(filename):
    ''' generate a numpy array from a comma delimitted text file '''
    return np.genfromtxt(filename, delimiter=",")

def celsius(a):
    ''' return column 1 (celsius) of a column major numpy array '''
    return a[:,1::]

def farenheit(a):
    ''' return column 0 (farenheit) of a cloumn major numpy array '''
    return a[:,:1:]

def plot(tu,tc,tp,bit):
    ''' plot numpy arrays '''
    ##temperature data
    fig =plt.figure()
    if bit == 0:
        plt.title('')
        plt.xlabel('temperature farenheit')
        plt.ylabel('temperature celsius')
        plt.plot(tu, tc,'o')
        #becomes a line with 13000 plotpoints
        #plt.plot(t_u.numpy(), t_c.numpy())  #i like the output of this as well
    if bit == 1:
        plt.title('???????')
        plt.xlabel("Temperature (°Fahrenheit)")
        plt.ylabel("Temperature (°Celsius)")
        #  we’re plotting the 
        # raw unknown values
        plt.plot(t_p.detach().numpy())

    plt.show(block=False)
    plt.pause(2) # Pause for interval seconds.
    #input("hit[enter] to end.")
    plt.close('all')

def model(tu,w,b):
    ''' linear model '''
    return w*tu+b

# tp-> t predicted
def loss_f(tp,tc):
    ''' return the mean of the squared differences of parrallel arrays '''
    squared_diffs=(tp-tc)**2
    return squared_diffs.mean()

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

# the name in main lies mainly on this plane
F = gen_fareneheit()
F2C()
csv(F)
A=nparray('degrees.csv')
print(A)
t_c=celsius(A)
t_u=farenheit(A)
tc_t=torch.tensor( t_c )
tu_t=torch.tensor( t_u )
n_validate=training(tc_t.shape[0])
shuffled_indices=shuffle(tc_t.shape[0])
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

plot(t_u,t_c,0,0)
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
t_p = model(tuvalidate_normt, *params)
# could use a little help here
plot(0,0,t_p,1)