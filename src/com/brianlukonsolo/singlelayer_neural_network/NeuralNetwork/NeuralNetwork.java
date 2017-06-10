package com.brianlukonsolo.singlelayer_neural_network.NeuralNetwork;

import com.brianlukonsolo.singlelayer_neural_network.Network.Network;
import com.brianlukonsolo.singlelayer_neural_network.NeuronTypes.InputNeuron;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.InputMismatchException;
import java.util.Scanner;

/**
 * Created by Brian Lukonsolo on 29/05/2017.
 */
public class NeuralNetwork {
    public static void main(String[] args) {
        //-------------------------------------------------------------------------------------------------[ User input ]
        Scanner userInput = new Scanner(System.in);

        //Read user input: Let the user enter a target output, learning rate and acceptable error level
        System.out.println("\n[0101010101010101010101010101010101010101010101010101010101010101010101]");
        System.out.println("[                      <><><><><><>^^<><><><><><>                      ]");
        System.out.println("[                  <><>SINGLE LAYER NEURAL NETWORK<><>                 ]");
        System.out.println("[              <><><><><><><><><><><><><><><><><><><><><>              ]");
        System.out.println("[          <><><><><><><><><><><>      <><><><><><><><><><><>          ]");
        System.out.println("[      <><><><><><><><><><><><>    00    <><><><><><><><><><><><>      ]");
        System.out.println("[      <><><><><><><><><><><><><>      <><><><><><><><><><><><><>      ]");
        System.out.println("[<><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>]");
        System.out.println("[0101010101010101010101010101010101010101010101010101010101010101010101]");
        System.out.println("[                                                                      ]");
        System.out.println("[                           [ Version 1.0 ]                            ]");
        System.out.println("[                      [ Completed: 10/06/2017 ]                       ]");
        System.out.println("[                     [ Author: Brian Lukonsolo ]                      ]");
        System.out.println("[                                                                      ]");
        System.out.println("[0101010101010101010101010101010101010101010101010101010101010101010101]\n");
        System.out.println("\n>>__________________________CONFIGURATION START__________________________<<");
        System.out.println("[Hint] During configuration, you may enter any number whether integer or decimal.\n");

        System.out.println("[Configuration] ---> Enter input 1: ");
        double input1 = userInput.nextDouble();
        System.out.println("[Configuration] ---> Enter input 2: ");
        double input2 = userInput.nextDouble();
        System.out.println("[Configuration] ---> Enter target output 1: ");
        double targetOutput1 = userInput.nextDouble();
        System.out.println("[Configuration] ---> Enter target output 2: ");
        double targetOutput2 = userInput.nextDouble();
        System.out.println("[Configuration] ---> Would you like to configure additional options? \n    Options >> ([1 = yes] [any other number = no])");
        int choice = userInput.nextInt();

        //Initialise advanced options with default values
        double learningRate = 40.5;
        double epochLimit = 10000;
        double acceptableDeviation = 0.01;

        if (choice == 1) {
            System.out.println("\n>>_____________________ADVANCED CONFIGURATION START______________________<<\n");
            System.out.println("[Advanced Configuration] ---> Enter a learning rate: [[ hint - the learning rate determines how quickly the neural network will find solutions ]]");
            learningRate = userInput.nextDouble();
            System.out.println("[Advanced Configuration] ---> Enter an acceptable deviation [[hint - this is a small number representing the amount of decimal point deviation you are willing to accept in your target outputs ]]");
            acceptableDeviation = userInput.nextDouble();
            System.out.println("[Advanced Configuration] ---> Enter an epoch limit: [[ hint - this is the maximum number of iterations the neural network can go through before giving up ]]");
            epochLimit = userInput.nextDouble();
        }

        //ArrayList of inputs
        ArrayList<InputNeuron> input_layer = new ArrayList<>();
        input_layer.add(new InputNeuron(input1));
        input_layer.add(new InputNeuron(input2));
        //-----INPUT NEURONS
        ArrayList<Double> inputsForHiddenNeurons = new ArrayList<>();
        for (InputNeuron inputNeuron : input_layer) {
            inputsForHiddenNeurons.add(inputNeuron.getInputValue());
        }

        //ArrayList of target outputs
        ArrayList<Double> targetOutputsList = new ArrayList<>();
        targetOutputsList.add(targetOutput1);
        targetOutputsList.add(targetOutput2);

        //-------------------------------------------------------------------------------------------------[ Run network ]
        //Instantiate the full network and pass it the input neurons arrayList as inputs + target outputs
        //Neural network accepts --> (inputsArrayList, targetOutputsArrayList)
        System.out.println("\n[ Configuration complete ] >> Press Enter to run the Neural Network: ");
        userInput.nextLine();
        System.out.println("\n>>____________________________TRAINING START_____________________________<<\n");
        //If user pressed enter, the network will be instantiated and will run
        Network neuralNetwork = new Network(inputsForHiddenNeurons, targetOutputsList);

        //:::::::: RUN THE NEURAL NETWORK ::::::::::::
        //Additional configuration of the network
        neuralNetwork.setLearningRate(learningRate);
        neuralNetwork.setAcceptableDeviation(acceptableDeviation);
        neuralNetwork.setEpochLimit(epochLimit);

        //Begin the learning process
        neuralNetwork.learn();

    }
}
