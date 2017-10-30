# A Keras like implementation in cuda with an implementation on half precision to improve performances

## Examples

Model M();
Model.add(10, 3);     // Add an input layer of size 3 with 10 neurons
Model.add(20, "relu") // Add a hiden layer of size 20 with a relu activation function
Model.add_max_POOL()  // Add a max pool layer of size 2*2

Model.train(X, y, 20, 0.1) // Train the model with 20 epochs and lr=0.1

Model.forward(X) // Forward propagation

## Examples

You can find an mnist example by compiling the main.cc file:

$ cd cpu/; make;

## Requirements
cuda, amrmadillo, cubla
cxx = c++14
