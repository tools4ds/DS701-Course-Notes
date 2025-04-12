# NN II and III Outline

## NN I Recap

* Applications of NNs
* Intuition on loss functions for model fitting -- convex and nonconvex
* Gradient descent intuition
* Derivatives refresher (slope, gradient)
* Gradient descent on linear regression model
* Challenges in gradient descent -- learning rate...
* Complex loss surfaces of NNs
* biological and artificial neurons
* first glimpse at a MLP/FCN


## NN II Outline

* Artificial Neuron -- picture
* Neuron -- scalar equation
* Neuron -- vector equation

Individual weights are $\omega$ and matrices of weights are $\Omega$.
Each neuron has its own set of weights for each input.

$\beta$ are vector of biases. Each neuron has it's own biases

* Shallow network -- 1 hidden layer with 4 neurons, single regression output
    * Show picture, 
    * then show matrix equation
    * define loss function...

* define as $loss(h_1(a_0(h_0(x))))$
    * linear part of hidden -- $h_0(x)$
    * activation function $a_0(.)$
    * linear part of output $h_1(.)$
    * loss function loss(.), say is just MSE

We want to each parameter by the negative of the partial derivative

* write the equations in terms of matrix values

* then we want the partial derivative of each to update the parameters

* Introduce the chain rule

* show a single neuron with single input
* show as compute graph

* show torchviz with numbers?
