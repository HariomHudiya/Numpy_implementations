# Since the entire file is only numpy implementation
# All the dependenct functions for activation are also implemented  and not imported from some library
import numpy as np

def softmax(x)  -> "Used for activation":
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))






# parameters Wax,Waa,Wya,ba,by
# Wax is used to take in x and predict a_prev
# Waa is used to take in a_prev and predict a_next
# Wya is used to take in a_next and predict y
# Note a_next denotes current activation


na = 5  # Number of units in the vector used to represent the hidden state at a given time_step
m = 10  # batch_size
nx = 3  # Number of units in the vector used to represent input(word) --> on-hot representation of words = Vocabulary size


np.random.seed(1)
# To store temporary parameters we'll make a dict , key -> Parameter_name: Value -> Para_value
parameters_tmp = {}

# Random initialisation of Parameters

Waa = np.random.randn(5,5)   # 5,5 dimension random initaialisation
Wax = np.random.randn(5,3)   # 5,3 dimension random initaialisation
Wya = np.random.randn(2,5)   # 2,5 dimension random initaialisation
ba = np.random.randn(5,1)
by = np.random.randn(2,1)

# Adding Values to dict
parameters_tmp['Waa'] = Waa
parameters_tmp['Wax'] = Wax
parameters_tmp['Wya'] = Wya
parameters_tmp['ba'] = ba
parameters_tmp['by'] = by

# To make our first prediction we'll also need
# 1 ) xt -- > Input at current timestep
# 2 ) a_prev -- > Activation at previous timestep

# Random initialisation

xt_tmp = np.random.randn(3,10)
a_prev_tmp = np.random.randn(5,10)

# Forward Pass Making a prediction at a given time_step

def rnn_forward(xt, a_prev, parameters):
    print("Making a forward pass in single Temporal Dimension")

    # Retriving parameters
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']

    # Computing Activation a_next

    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    yt_pred = softmax(np.dot(Wya, a_next) + by)

    # Caching Values as they'll be needed in backprop
    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache

# Executing the function
a_next_tmp, yt_pred_tmp, cache_tmp = rnn_forward(xt_tmp, a_prev_tmp, parameters_tmp)

# Debugging info
# print("a_next[4] = \n", a_next_tmp[4])
# print("a_next.shape = \n", a_next_tmp.shape)
# print("yt_pred[1] =\n", yt_pred_tmp[1])
# print("yt_pred.shape = \n", yt_pred_tmp.shape)


########## Rolling out in Temporal_Dimension #############

def rnn_forward_full_pass(X, a0, parameters):

    """
    X -- > Input Tensor (nx,m,Tx)
    a0 -- > Initial hidden state (na,m)
    parameters --> Python Dict
    """
    print("---------------------------------")
    print("Executing the Full Forward Pass")
    # Cache to store list of caches at each time step
    caches = []

    # Retriving Dimensions
    nx, m, Tx = X.shape
    ny,na = parameters['Wya'].shape

    # Initialise activation and prediction for "Entire" Network to zeero
    A = np.zeros((na,m,Tx))
    Y_pred = np.zeros((ny,m,Tx))

    # Initialise a_next
    a_next = a0

    # Loop over all time-steps

    for t in range(Tx):
        a_next,yt_pred,cache = rnn_forward(X[:,:,t],a_next,parameters)
        A[:,:,t] = a_next
        Y_pred[:,:,t] = yt_pred
        caches.append(cache)


    print("--------------------------------")
    # store values needed for backward propagation in cache
    caches = (caches, X)

    return A,Y_pred,caches

####### Calling the Function #########

np.random.seed(1)

nx = 3   # Number of units used to represent a input word(x)
na = 5   # Number of units used to represent activation at a given time-step
ny = 2   # Number of units used to represent an output
m = 10   # Batch_size
Tx = 4   # Temporal_dimension

x_tmp = np.random.randn(nx, m, Tx)
a0_tmp = np.random.randn(na, m)
parameters_tmp = {}
parameters_tmp['Waa'] = np.random.randn(na,na)
parameters_tmp['Wax'] = np.random.randn(na,nx)
parameters_tmp['Wya'] = np.random.randn(ny,na)
parameters_tmp['ba'] = np.random.randn(na,1)
parameters_tmp['by'] = np.random.randn(ny,1)

a_tmp, y_pred_tmp, caches_tmp = rnn_forward_full_pass(x_tmp, a0_tmp, parameters_tmp)

# Debugging Info
# print("a[4][1] = \n", a_tmp[4][1])
# print("a.shape = \n", a_tmp.shape)
# print("y_pred[1][3] =\n", y_pred_tmp[1][3])
# print("y_pred.shape = \n", y_pred_tmp.shape)
# print("caches[1][1][3] =\n", caches_tmp[1][1][3])
# print("len(caches) = \n", len(caches_tmp))



########################## LSTM CELL ###################
print("LSTM Cell Operation Starts here.")

# Hidden _state != Cell _State
# Forget Gate (ft)

# The "forget gate" is a tensor containing values that are between 0 and 1.
# If a unit in the forget gate has a value close to 0,
# the LSTM will "forget" the stored state in the corresponding unit of the previous cell state.
# If a unit in the forget gate has a value close to 1,
# the LSTM will mostly remember the corresponding value in the stored state.

# Wf -- > Weights of Forget Gate (shared parameter)
# bf -- > Forget Gate Bias        (shared parameter)
# ft -- > Value of Forget gate at time step t  (Different for each time step)

# Candidate Value (cct) <Candidate Cell State at time t>

# The candidate value is a tensor containing information from current time step.
# This information may be stored in the current cell state
# Which parts of the candidate value gets passed Depends on Update Gate.

# Update Gate (it)
# Update Gate is used to decide what aspects of the candidate to add to current cell
# When a unit in the update gate is close to 1,
# it allows the value of the candidate  to be passed onto the cell state ct
# When a unit in the update gate is close to 0,
# it prevents the corresponding value in the candidate from being passed onto the cell state (current state)
# Wi -- > Update Gate Weight
# bi -- > Update Gate bias
# it --> Value of the update Gate at time step t

# Cell State (ct)

# The Cell state is the "memory" that gets passed onto future time steps.

# C -- > (na,m,Tx)
# c_next -- > (na,m)
# c_prev -- > (na,m)

# Output Gate (ot)

# The output gate decides what gets sent as the prediction(output) of the timestep

# Wo -- > Output Gate Weight
# bo -- > Output Gate Bias
# ot -- > Output Gate Value at time step t

# Hidden State (at)

# The Hidden state gets passed to LSTM's next time step
# It is used to determine values of the three gates fot next timestep
# It is also used for prediction at current time step

# A -- > (na,m,Tx)
# a_prev -- > (na,m)
# a_next -- > (na,m)

# Prediction (y_pred)

# The Prediction of the given time_step

# Y_pred -- > (ny,m,Tx)
# yt_pred -- > (ny,m)

def lstm_cell_forward(xt,a_prev,c_prev,parameters):
    """

    :param xt:  input data at time step t , (nx,m)
    :param a_prev: Hidden state at time step t-1 , (na,m)
    :param c_prev: Memory state at time step t-1 , (na,m)
    :param paramters: Dictonary containing

                        Wf -- Weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        bf -- Bias of the forget gate, numpy array of shape (n_a, 1)
                        Wi -- Weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        bi -- Bias of the update gate, numpy array of shape (n_a, 1)
                        Wc -- Weight matrix of the first "tanh", numpy array of shape (n_a, n_a + n_x)
                        bc --  Bias of the first "tanh", numpy array of shape (n_a, 1)
                        Wo -- Weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        bo --  Bias of the output gate, numpy array of shape (n_a, 1)
                        Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    :return:

    a_next -- next hidden state, of shape (n_a, m)
    c_next -- next memory state, of shape (n_a, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, c_next, a_prev, c_prev, xt, parameters)

    Note: ft/it/ot stand for the forget/update/output gates, cct stands for the candidate value (c tilde),
          c stands for the cell state (memory)
    """

    print("LSTM forward pass for single temporal Dimension")

    # Retrieve parameters from "parameters"
    Wf = parameters["Wf"]  # forget gate weight
    bf = parameters["bf"]
    Wi = parameters["Wi"]  # update gate weight (notice the variable name)
    bi = parameters["bi"]  # (notice the variable name)
    Wc = parameters["Wc"]  # candidate value weight
    bc = parameters["bc"]
    Wo = parameters["Wo"]  # output gate weight
    bo = parameters["bo"]
    Wy = parameters["Wy"]  # prediction weight
    by = parameters["by"]



    # Concating a_prev and xt
    # Vertical Concatination
    concat = np.concatenate((a_prev,xt), axis = 0)

    # Compute values for ft (forget gate), it (update gate),
    # cct (candidate value), c_next (cell state),
    # ot (output gate), a_next (hidden state)
    ft = sigmoid(np.dot(Wf, concat) + bf)  # forget gate
    it = sigmoid(np.dot(Wi, concat) + bi)  # update gate
    cct = np.tanh(np.dot(Wc, concat) + bc)  # candidate value
    c_next = (ft * c_prev) + (it * cct)  # cell state
    ot = sigmoid(np.dot(Wo, concat) + bo)  # output gate
    a_next = ot * np.tanh(c_next)  # hidden state

    # Making prediction at a time_step
    yt_pred = softmax(np.dot(Wy,a_next) + by)

    # store values needed for backward propagation in cache
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache

######### Calling LSTM Function ##########

np.random.seed(1)
xt_tmp = np.random.randn(3,10)
a_prev_tmp = np.random.randn(5,10)
c_prev_tmp = np.random.randn(5,10)
parameters_tmp = {}
parameters_tmp['Wf'] = np.random.randn(5, 5+3)
parameters_tmp['bf'] = np.random.randn(5,1)
parameters_tmp['Wi'] = np.random.randn(5, 5+3)
parameters_tmp['bi'] = np.random.randn(5,1)
parameters_tmp['Wo'] = np.random.randn(5, 5+3)
parameters_tmp['bo'] = np.random.randn(5,1)
parameters_tmp['Wc'] = np.random.randn(5, 5+3)
parameters_tmp['bc'] = np.random.randn(5,1)
parameters_tmp['Wy'] = np.random.randn(2,5)
parameters_tmp['by'] = np.random.randn(2,1)

a_next_tmp, c_next_tmp, yt_tmp, cache_tmp = lstm_cell_forward(xt_tmp, a_prev_tmp, c_prev_tmp, parameters_tmp)
# print("a_next[4] = \n", a_next_tmp[4])
# print("a_next.shape = ", a_next_tmp.shape)
# print("c_next[2] = \n", c_next_tmp[2])
# print("c_next.shape = ", c_next_tmp.shape)
# print("yt[1] =", yt_tmp[1])
# print("yt.shape = ", yt_tmp.shape)
# print("cache[1][3] =\n", cache_tmp[1][3])
# print("len(cache) = ", len(cache_tmp))


def lstm_forward(X,a0,parameters):
    """

    :param X -- > Input data of shape (n_x,m,T_x):
    :param a0 -- >Initial hidden state (n_a , m):
    :param parameters -- > Python Dictionary:
                Wf -- Weight Matrix of Forget Gate, (na, na + nx)
                bf -- Bias of the Forget Gate , (na,1)
                Wi -- Weight Matrix of Update Gate , (na,na+nx)
                bi -- Bias of Update Gate, (na,1)
                Wo -- Wieght Matrix of Output Fate , (na,na+nx)
                bo -- Bias of Output Gate,(na,1)
                Wc -- Weight matrix to calculate cct, (na,na+nx)
                bc -- Bias to calculate cct , (na,1)
                Wy -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    :return:
    a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
    y -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    c -- The value of the cell state, numpy array of shape (n_a, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of all the caches, x)
    """
    print("----------------------------")
    print("LSTM FULL forward started")

    # Initialise caches , to store list of cache
    caches = []

    #Retriving Parameters
    Wy = parameters['Wy']
    nx,m,Tx = X.shape
    ny,na = Wy.shape

    # Initialise 'A','C' and 'Y' with zeros
    # We will populate these
    A = np.zeros((na,m,Tx))
    C = np.zeros((na,m,Tx))
    Y = np.zeros((ny,m,Tx))

    # Initalise current cell_state and hidden_state

    a_next = a0
    c_next = np.zeros((na,m))

    # looping through time

    for t in range(Tx):

        # Current Input

        xt = X[:,:,t]

        # Update the current cell state
        # Update the current hidden state
        # Compute the current prediction

        a_next,c_next,yt,cache = lstm_cell_forward(xt, a_next, c_next, parameters)

        # Updating Matrices A,C,Y

        A[:,:,t] = a_next
        C[:,:,t] = c_next
        Y[:,:,t] = yt

        caches.append(cache)

    # store values needed for backward propagation in cache
    caches = (caches, X)
    print("---------------")

    return A, Y, C, caches


############# Calling Function full LSTM ################
np.random.seed(1)
x_tmp = np.random.randn(3,10,7)
a0_tmp = np.random.randn(5,10)
parameters_tmp = {}
parameters_tmp['Wf'] = np.random.randn(5, 5+3)
parameters_tmp['bf'] = np.random.randn(5,1)
parameters_tmp['Wi'] = np.random.randn(5, 5+3)
parameters_tmp['bi']= np.random.randn(5,1)
parameters_tmp['Wo'] = np.random.randn(5, 5+3)
parameters_tmp['bo'] = np.random.randn(5,1)
parameters_tmp['Wc'] = np.random.randn(5, 5+3)
parameters_tmp['bc'] = np.random.randn(5,1)
parameters_tmp['Wy'] = np.random.randn(2,5)
parameters_tmp['by'] = np.random.randn(2,1)

a_tmp, y_tmp, c_tmp, caches_tmp = lstm_forward(x_tmp, a0_tmp, parameters_tmp)
# print("a[4][3][6] = ", a_tmp[4][3][6])
# print("a.shape = ", a_tmp.shape)
# print("y[1][4][3] =", y_tmp[1][4][3])
# print("y.shape = ", y_tmp.shape)
# print("caches[1][1][1] =\n", caches_tmp[1][1][1])
# print("c[1][2][1]", c_tmp[1][2][1])
# print("len(caches) = ", len(caches_tmp))


############## Backwardpass ###########################

# TO DO :-
# Implement Backward pass using Gradient tape (tensorflow)

#######################################################

def rnn_backward(X, Y, parameters, cache):
    # Initialize gradients as an empty dictionary
    gradients = {}

    # Retrieve from cache and parameters
    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']

    # each one should be initialized to zeros of the same dimension as its corresponding parameter
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])

    ### START CODE HERE ###
    # Backpropagate through time
    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t - 1])
    ### END CODE HERE ###

    return gradients, a


def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next']  # backprop into h
    daraw = (1 - a * a) * da  # backprop through tanh nonlinearity
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients