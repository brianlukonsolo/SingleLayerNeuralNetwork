# SingleLayerNeuralNetwork
A personal Java project in which I experiment with creating single layer artificial neural networks

- This project is a neural network with an input layer, a hidden layer and an output layer.

-- Visualizing the neural network

   Input 1 -->  o      o      o  <-- Output 1

   Input 2 -->  o      o      o  <-- Output 2


  >> To start the program, open the project in IntelliJ IDE and run the NeuralNetwork class.

  >> You will be presented with a configuration setup and each option will be shown to you one by one.
  >> You will be asked for:

    - Input 1 - This is the first input that the network will use
    - Input 2 - This is the second input that the network will use
    - Target output 1 - This is the number that the network will learn to achieve for output 1 of the network
    - Target output 2 - This is the number that the network will learn to achieve for output 2 of the network

  >> Next you will be asked if you want to perform additional configuration
  >> You will be asked for:

    - Learning rate - This is a multiplier value that can speed up or slow down the learning of the network
    - Acceptable deviation - Learning will be considered successful if the actual output is greater than ( actual output - acceptable deviation ) and less than (actual output + acceptable deviation ).
    - Epoch limit - The maximum number of iterations the network should attempt during learning. This is a failsafe and prevents infinite looping if the learning fails to reach a solution.

  >> Finally the network will run and data will be displayed for each epoch.

  Thank you for trying my program! :)

