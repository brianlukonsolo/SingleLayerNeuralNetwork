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

      /*
      //add two hidden neurons to the hidden layer----------------------------------------HIDDEN NEURON LAYER
      ArrayList<HiddenNeuron> hidden_layer = new ArrayList<>();
      hidden_layer.add(new HiddenNeuron(inputsForHiddenNeurons, 0.35));
      hidden_layer.add(new HiddenNeuron(inputsForHiddenNeurons, 0.35));
//Giving one of the hidden neurons some test weights

      //====================TESTING ZONE=====================================================================########
      //Hidden Neuron 1
      HiddenNeuron test = new HiddenNeuron(inputsForHiddenNeurons, 0.35);
      test.setWeightsOfInputs(new ArrayList<Double>(Arrays.asList(0.15,0.20))); //The weights of the synapses
      double ans = test.calculateNetSum();
      double output1 = test.calculateSigmoidOutput(ans);

      //Hidden Neuron 2
      HiddenNeuron test2 = new HiddenNeuron((inputsForHiddenNeurons), 0.35);
      test2.setWeightsOfInputs(new ArrayList<>(Arrays.asList(0.25, 0.30)));
      double ans2 = test2.calculateNetSum();
      double output2 = test2.calculateSigmoidOutput(ans2);

      //Add the neurons to hidden layer
      hidden_layer.clear();
      hidden_layer.add(test);
      hidden_layer.add(test2);
      //=====================END OF TESTING ZONE=============================================================########




      //-----ARRAYLIST OF THE INPUTS TO PASS TO THE OUTPUT LAYER NEURONS
      ArrayList<Double> inputsForOutputNeurons = new ArrayList<>();
      for(HiddenNeuron hiddenNeuron: hidden_layer){
          inputsForOutputNeurons.add(hiddenNeuron.getSigmoidOutputOfTheHiddenNeuron());
      }


      //add two output neurons to the output layer----------------------------------------OUTPUT NEURON LAYER
      ArrayList<OutputNeuron> output_layer = new ArrayList<>();
      OutputNeuron out1 = new OutputNeuron(inputsForOutputNeurons, 0.60);
      out1.setWeightsOfOutputsFromHiddenLayer(new ArrayList<>(Arrays.asList(0.40, 0.45)));
      double ns1 = out1.calculateNetSum();
      double finalOut = out1.calculateSigmoidOutput(ns1);

      OutputNeuron out2 = new OutputNeuron(inputsForOutputNeurons, 0.60);
      out2.setWeightsOfOutputsFromHiddenLayer(new ArrayList(Arrays.asList(0.50, 0.55)));
      double ns2 = out2.calculateNetSum();
      double finalOut2 = out2.calculateSigmoidOutput(ns2);
*/

      //Instatiate the full network and pass it the input neurons arrayList as inputs
      Network neuralNetwork = new Network(inputsForHiddenNeurons);

      System.out.println("----------------------------ABOUT TO FOWARD-PROPAGATE------------------------------------");
      //Forward pass
      neuralNetwork.fowardPropagate();

      //neuralNetwork.startLearning(inputsForHiddenNeurons);

     // for(InputNeuron d: neuralNetwork.getInputNeuronsList()) {
     //     System.out.println("Network input: " + d.getInputValue());
     // }


  }
}
