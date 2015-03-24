# MultilayerPerceptron

**Simple java ANN Multilayer Perceptrom (MLP).**

Artificial Neural Network based on Multilayer Perceptron using [EJML library](https://code.google.com/p/efficient-java-matrix-library/) for matrix management (the *jar* file is available [here](https://efficient-java-matrix-library.googlecode.com/files/ejml-0.24.jar)).

Algorithm based on **Chapter 3** and **Appendix A** of the book **Practical Neural Networks and Genetic Algorithms**. If you want to learn more about this stuff, I recomend you to take a look to the [book](https://github.com/dsaiztc/MultilayerPerceptron#practical-neural-networks-and-genetic-algorithms).

## How to use
Currently the *threeshold activation* is not available (that's a bug pendent to fix). But you can use *sigmoid activation* for your purposes. If you take a look to the source code, for using a MLP the only thing you need to do is to create an *ANN* object:

````
// 2 inputs, 3 hidden neurons, 1 output, 1 hidden layer, 1.0 learning rate, activation function
ANN myNeuralNetwork = new ANN(2, 3, 1, 1, 1.0, ANN.ACTIVATION_SIGMOID);
````

An then to train the neural network just call *train* method:

````
// Input vector, target vector, minimum operations, maximum error
myNeuralNetwork.train(input, target, 10000, 0.01);
````

The format of the input labeled data for training should be:

````
-- --                                 -- --
| input1[0]   input2[0]   ... inputN[0]   | 
| input1[1]   input2[1]   ... inputN[1]   | 
|    |            |               |       |  
| input1[M-1] input2[M-1] ... inputN[M-1] |
-- --                                 -- --
````

Where as you can see, in the double array:

````
[i][j] i->Input j->Number of input
````

After the nerual network has been trained, you will be able to classify your data. For this purpose, you can use the *classify* function:

````
output = myNeuralNetwork.classify(input);
````

And that's all!

## Contributions
Please, feel free to collaborate or give me some feedback. Hope this code help you!

## Practical Neural Networks and Genetic Algorithms
It is available online on [Robert Gordon University Aberdeen](http://www.rgu.ac.uk/):

- [Title page](https://www4.rgu.ac.uk/files/ACF58D4.pdf).
- [Contents](http://www4.rgu.ac.uk/files/ACF58D0.pdf).
- [Chapter 1 - An Introduction to Neural Networks](http://www4.rgu.ac.uk/files/chapter1%20-%20intro.pdf).
- [Chapter 2 - Artificial Neural Networks](https://www4.rgu.ac.uk/files/chapter2%20-%20intro%20to%20ANNs.pdf).
- [Chapter 3 - The Back Propagation Algorithm](https://www4.rgu.ac.uk/files/chapter2%20-%20intro%20to%20ANNs.pdf).
- [Chapter 4 - Some illustrative applications of feed-forward networks](http://www4.rgu.ac.uk/files/chapter4%20-applications.pdf).
- [Chapter 5 - Pre-processing input data](http://www4.rgu.ac.uk/files/chapter5%20-%20pre-processing.pdf).
- [Chapter 6 - Network layers and size](https://www4.rgu.ac.uk/files/chapter6%20-%20network%20size.pdf).
- [Chapter 7 - Hopfield and recurrent networks](https://www4.rgu.ac.uk/files/chapter7-hopfield.pdf).
- [Chapter 8 - Competitive networks](http://www4.rgu.ac.uk/files/chapter8%20-%20competitive.pdf).
- [Chapter 9 - Time dependant neurons](https://www4.rgu.ac.uk/files/chapter9%20-%20spiky.pdf).
- [Chapter 10 - Implementing Neural Nets](https://www4.rgu.ac.uk/files/chapter10%20-%20implementing%20ANNs.pdf).
- [Chapter 11 - An introduction to Evolution](https://www4.rgu.ac.uk/files/chapter11%20-%20intro%20to%20evolution.pdf).
- [Chapter 12 - The Genetic Algorithm](https://www4.rgu.ac.uk/files/chapter12%20-%20GAs.pdf).
- [Chapter 13 - Some applications of Genetic Algorithms](http://www4.rgu.ac.uk/files/chapter13%20-%20applications.pdf).
- [Chapter 14 - Additions to the basic GA](https://www4.rgu.ac.uk/files/chapter14%20-%20extra%20GA.pdf).
- [Chapter 15 - Applying Genetic Algorithms to Neural Networks](https://www4.rgu.ac.uk/files/chapter15%20-%20eanns.pdf).
- [Chapter 16 - Evolutionary Programming and Evolutionary Strategies](http://www4.rgu.ac.uk/files/chapter16%20-%20ESEP.pdf).
- [Chapter 17 - Evolutionary Algorithms in advanced AI systems](https://www4.rgu.ac.uk/files/chapter17%20-%20advanced%20ai.pdf).
- [Appendix A - Appendix A Tips for Programmers](http://www4.rgu.ac.uk/files/ACF58BB.pdf).
- [References](http://www4.rgu.ac.uk/files/ACF58D2.pdf).
