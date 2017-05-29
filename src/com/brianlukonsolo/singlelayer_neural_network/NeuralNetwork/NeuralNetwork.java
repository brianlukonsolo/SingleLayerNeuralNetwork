package com.brianlukonsolo.singlelayer_neural_network.NeuralNetwork;

import com.brianlukonsolo.singlelayer_neural_network.Network.Network;
import com.brianlukonsolo.singlelayer_neural_network.NeuronTypes.HiddenNeuron;
import com.brianlukonsolo.singlelayer_neural_network.NeuronTypes.InputNeuron;
import com.brianlukonsolo.singlelayer_neural_network.NeuronTypes.OutputNeuron;

import java.util.ArrayList;
import java.util.Arrays;

/**
 * Created by lordmcbrian on 29/05/2017.
 */
public class NeuralNetwork {
  public static void main(String[] args){
      //The NeuralNetwork contains all the neurons
      System.out.println("Neural network is alive");


      //Create an input-layer, hidden-layer and output-layer for the network to use

      //add two inputs to the input layer------------------------------------------------INPUT NEURON LAYER
      ArrayList<InputNeuron> input_layer = new ArrayList<>();
      //Add the input neurons to the list of input neurons
      input_layer.add(new InputNeuron(0.05));
      input_layer.add(new InputNeuron(0.10));

      //-----ARRAYLIST OF THE INPUTS TO PASS TO THE HIDDEN LAYER NEURONS
      ArrayList<Double> inputsForHiddenNeurons = new ArrayList<>();
      for(InputNeuron inputNeuron: input_layer){
          inputsForHiddenNeurons.add(inputNeuron.getInputValue());
      }

      //Instatiate the full network and pass it the input neurons arrayList as inputs
      Network neuralNetwork = new Network(inputsForHiddenNeurons);

      System.out.println("\n----------------------------THE FORWARD-PASS------------------------------------");
      //Forward pass
      neuralNetwork.fowardPropagate();

      //TODO: neuralNetwork.backpropagate

      //TODO: neuralNetwork.updateAllWeights

  }
}
