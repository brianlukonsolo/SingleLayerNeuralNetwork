package com.brianlukonsolo.singlelayer_neural_network.NeuralNetwork;

import com.brianlukonsolo.singlelayer_neural_network.Network.Network;
import com.brianlukonsolo.singlelayer_neural_network.NeuronTypes.HiddenNeuron;
import com.brianlukonsolo.singlelayer_neural_network.NeuronTypes.InputNeuron;
import com.brianlukonsolo.singlelayer_neural_network.NeuronTypes.OutputNeuron;

import java.util.ArrayList;

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

      //add two hidden neurons to the hidden layer----------------------------------------HIDDEN NEURON LAYER
      ArrayList<HiddenNeuron> hidden_layer = new ArrayList<>();
      hidden_layer.add(new HiddenNeuron());
      hidden_layer.add(new HiddenNeuron());

      //add two output neurons to the output layer----------------------------------------OUTPUT NEURON LAYER
      ArrayList<OutputNeuron> output_layer = new ArrayList<>();
      output_layer.add(new OutputNeuron());
      output_layer.add(new OutputNeuron());


      //Instatiate the full network and pass it the input neurons arrayList as inputs
      Network neuralNetwork = new Network(input_layer, hidden_layer, output_layer);
      //Run the network
      neuralNetwork.startLearning();

      for(InputNeuron d: neuralNetwork.getInputNeuronsList()) {
          System.out.println("Network input: " + d.getInputValue());
      }


  }
}
