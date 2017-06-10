package com.brianlukonsolo.singlelayer_neural_network.NeuronTypes;

import java.util.ArrayList;

/**
 * Created by Brian Lukonsolo on 29/05/2017.
 */

//Each hidden neuron has both forward and backward connections in the network
// InputNeuron --> [O] ------ HiddenNeuron [[[o]]] -------- OutputNeuron [o]

//################################################################################
//Each hidden neuron has a list of weights corresponding with the number of inputs
//For example: if the Inputs = {input1, input2}
//        therefore, weights = {0.1, 0.34}
//
//Further example: Inputs = {input1, input2, input3, input4}
//                weights = {0.2, 0.54, 0.22, 0.42}
//
//################################################################################

public class HiddenNeuron {
    private ArrayList<Double> inputsLayerValuesList = new ArrayList<>();
    private ArrayList<Double> weightsOfInputs = new ArrayList<>();
    private double bias = 0.35;
    private double netSumOfInputToTheHiddenNeuron;
    private double sigmoidOutputOfTheHiddenNeuron;

    //Constructor
    public HiddenNeuron(ArrayList<Double> inputs, double bias_neuron_value) {
        this.setInputsLayerValuesList(inputs);
        this.setBias(bias_neuron_value);
        //TODO: Generate random weights for each input!!!
    }

    public double calculateNetSum(ArrayList<Double> relevant_weights) {
        double netSum = bias;

        //Net sum of the inputs to the neuron multiplied by their weights
        for (int i = 0; i < getInputsLayerValuesList().size(); i++) {
            netSum = netSum + (getInputsLayerValuesList().get(i) * relevant_weights.get(i));
        }
        //Store the value for later use
        setNetSumOfInputToTheHiddenNeuron(netSum);

        return netSum;
    }

    public double calculateSigmoidOutput(double netInput) {
        //Sigmoid activation function
        double outputValue = (1 / (1 + Math.exp(-netInput)));
        //DEBUG
        System.out.println("[ HiddenNeuron Sigmoid function ]>>> Output of sigmoid: " + outputValue);
        //Store the value for later use
        setSigmoidOutputOfTheHiddenNeuron(outputValue);
        return outputValue;
    }

    public double fire(ArrayList<Double> relevant_weights) {
        double sigmoidOutput = calculateSigmoidOutput(calculateNetSum(relevant_weights));
        return sigmoidOutput;
    }

    //Get and Set
    public ArrayList<Double> getWeightsOfInputs() {
        return weightsOfInputs;
    }

    public void setWeightsOfInputs(ArrayList<Double> weightsOfInputs) {
        this.weightsOfInputs = weightsOfInputs;
    }

    public ArrayList<Double> getInputsLayerValuesList() {
        return inputsLayerValuesList;
    }

    public void setInputsLayerValuesList(ArrayList<Double> inputsLayerValuesList) {
        this.inputsLayerValuesList = inputsLayerValuesList;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getSigmoidOutputOfTheHiddenNeuron() {
        return sigmoidOutputOfTheHiddenNeuron;
    }

    public void setSigmoidOutputOfTheHiddenNeuron(double sigmoidOutputOfTheHiddenNeuron) {
        this.sigmoidOutputOfTheHiddenNeuron = sigmoidOutputOfTheHiddenNeuron;
    }

    public double getNetSumOfInputToTheHiddenNeuron() {
        return netSumOfInputToTheHiddenNeuron;
    }

    public void setNetSumOfInputToTheHiddenNeuron(double netSumOfInputToTheHiddenNeuron) {
        this.netSumOfInputToTheHiddenNeuron = netSumOfInputToTheHiddenNeuron;
    }
}
