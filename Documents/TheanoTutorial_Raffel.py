import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T


## Learning scalars and simple functions
#foo = T.scalar('foo')
#
#def square(x):
#    return x**2
#
#bar = square(foo)
#
#print type(bar)
#print bar.type
#print theano.pp(bar)
#
#f = theano.function([foo], bar)
#
#print f(3)
#print bar.eval({foo: 3})


## Learning how to use the Theano matrix, vector and built in functions for these
#A = T.matrix('A')
#x = T.vector('x')
#b = T.vector('b')
#y = T.dot(A,x) + b
#z = T.sum(A**2)
#
#linear_mix = theano.function([A, x, theano.Param(b, default=np.array([0, 0]))], [y, z])
#
#print linear_mix(np.array([[1, 2, 3], [4, 5, 6]]), np.array([1, 2, 3]), np.array([4, 5]))
#
#print linear_mix(np.array([[1, 2, 3], [4, 5, 6]]), np.array([1, 2, 3]))


## Shared variables
#shared_var = theano.shared(np.array([[1, 2], [3, 4]], dtype = theano.config.floatX))
#print shared_var.type()
#print shared_var.get_value()
#
#shared_var.set_value(np.array([[3, 4], [2, 1]], dtype=theano.config.floatX))
#print shared_var.type()
#print shared_var.get_value()
#
#shared_squared = shared_var**2
## Since shared_Var is shared, it's implicitly an input to a function using shared_squared!
#function_1 = theano.function([], shared_squared)
#print function_1()
#
#subtract = T.matrix('subtract')
## This function now updates the shared_var by subtracting with the input
#function_2 = theano.function([subtract], shared_var, updates = {shared_var: shared_var - subtract})
#
#function_2(np.array([[1, 1], [1, 1]]))
#print shared_var.get_value()
## One will see that this obviously also updates the result from function_1 
## as the shared_var has changed
#print function_1()

## Gradients
#foo = T.scalar('foo')
#too = T.scalar('too')
#soo = T.scalar('soo')
#bar = foo**2 + foo*soo**2
#
#bar_grad = T.grad(bar,foo)
#
#f = theano.function([foo, soo], bar_grad)
#print f(10, 10)
#
#A = T.matrix('A')
#x = T.vector('x')
#b = T.vector('b')
#y = T.dot(A,x) + b
#
#hes_x = T.dvector('hes_x')
#hes_bar = hes_x ** 2
#cost = hes_bar.sum()
#
#y_J = theano.gradient.jacobian(y,x)
#y_H = theano.gradient.hessian(hes_bar, hes_x)
#linear_mix_J = theano.function([A, x, b], y_J)
#
#print linear_mix_J(np.array([[9, 8, 7], [4, 5, 6]]), 
#                   np.array([1, 2, 3]),
#                   np.array([4, 5]))

#print y_H([10, 20, 30])

## Example: MLP
# The layer class takes initialization values and makes sure W and b are of floatX type.
# An MLP is defined as many of these layers. A layer of a neural network computes s(Wx + b)
# where s is a nonlinearity and x is the input vector

class Layer(object):
    def __init__(self, W_init, b_init, activation):
        n_output, n_input = W_init.shape
        assert b_init.shape == (n_output,)
        self.W = theano.shared(value=W_init.astype(theano.config.floatX),
                               name = 'W',
                               borrow=True)

        self.b = theano.shared(value=b_init.reshape(-1, 1).astype(theano.config.floatX),
                               name = 'b',
                               borrow = True,
                               broadcastable = (False, True))

        self.activation = activation

        self.params = [self.W, self.b]

    def output(self, x):
        lin_output = T.dot(self.W, x) + self.b
        return (lin_output if self.activation is None else self.activation(lin_output))


# The MLP class is basically just a container of layers since each layer contains most of its
# information. The most significant functionality it has is the computation of the output
# and consequently also the squared error estimate

class MLP(object):
    def __init__(self, W_init, b_init, activations):
        assert len(W_init) == len(b_init) == len(activations)
        
        self.layers = []
        for W, b, activation in zip(W_init, b_init, activations):
            self.layers.append(Layer(W, b, activation))

        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def output(self, x):
        for layer in self.layers:
            x = layer.output(x)
        return x

    def squared_error(self, x, y):
        return T.sum((self.output(x) - y)**2)


## Gradient descent

def gradient_updates_momentum(cost, params, learning_rate, momentum):
    assert momentum < 1 and momentum >= 0
    updates = []

    for param in params:
        param_update = theano.shared(param.get_value()*0., broadcastable = param.broadcastable)
        updates.append((param, param - learning_rate*param_update))
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost,param)))

    return updates

## Toy example using Gaussian-distributed clusters in 2d space.

#plotting a gaussian distributed cluster
np.random.seed(0)
N = 1000
y = np.random.random_integers(0, 1, N)
means = np.array([[-1, 1], [-1, 1]])
covariances = np.random.random_sample((2, 2)) + 1

X = np.vstack([np.random.randn(N)*covariances[0, y] + means[0, y], 
               np.random.randn(N)*covariances[1, y] + means[1, y]])

#plt.figure(figsize=(8, 8))
#plt.scatter(X[0, :], X[1, :], c=y, lw=.3, s=3, cmap=plt.cm.cool)
#plt.axis([-6, 6, -6, 6])
#plt.show()


# initializing and defining training function with theano

layer_sizes = [X.shape[0], X.shape[0]*2, 1] 

W_init = []
b_init = []
activations = []

for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
    W_init.append(np.random.randn(n_output, n_input))
    b_init.append(np.ones(n_output))

    activations.append(T.nnet.sigmoid)

mlp = MLP(W_init, b_init, activations)

mlp_input = T.matrix('mp_input')
mlp_target = T.vector('mlp_target')

learning_rate = 0.01
momentum = 0.9
cost = mlp.squared_error(mlp_input, mlp_target)

train = theano.function([mlp_input, mlp_target], cost, 
                        updates = gradient_updates_momentum(cost, 
                                                            mlp.params, 
                                                            learning_rate, 
                                                            momentum))

mlp_output = theano.function([mlp_input], mlp.output(mlp_input))

# executing training 
iteration = 0
max_iteration = 3

while iteration < max_iteration:
    current_cost = train(X, y)
    current_output = mlp_output(X)

    accuracy = np.mean((current_output > .5) == y)

    plt.figure(figsize = (8, 8))
    plt.scatter(X[0, :], X[1, :], c = current_output, 
                lw = .3, s = 3, cmap = plt.cm.cool, vmin = 0, vmax = 1)
    plt.axis([-6, 6, -6, 6])
    plt.title('Cost: {:.3f}, Accuracy: {:.3f}'.format(float(current_cost),accuracy))
    plt.show()
    iteration += 1
