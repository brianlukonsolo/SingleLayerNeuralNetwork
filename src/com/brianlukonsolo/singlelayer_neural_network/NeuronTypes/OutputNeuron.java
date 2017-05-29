package com.brianlukonsolo.singlelayer_neural_network.NeuronTypes;

import java.util.ArrayList;

/**
 * Created by lordmcbrian on 29/05/2017.
 */

//Each output neuron only has backward connections in the network, from right to left
// InputNeuron[O] ------ HiddenNeuron [o] -------- [[[o]]] <-- OutputNeuron

public class OutputNeuron {
    //The hidden layer's outputs are used as inputs in the output layer
    private ArrayList<Double> hiddenLayerOutputValuesList = new ArrayList<>();
    //The weights of the synapses between the hidden layer outputs and the output layer
    private ArrayList<Double> weightsOfOutputsFromHiddenLayer = new ArrayList<>();

    //Target and actual outputs
    private double targetOutput;
    private double actualOutput;

    //Error between target output and actual output
    private double outputError;
    private double bias;
    //Storages: will be used during the back-propagation algorithm
    private double sigmoidOutputOfTheOutputNeuron;
    private double netSumOfTheInputsToTheOutputNeuron;


    //Constructor
    public OutputNeuron(ArrayList<Double> outputs_of_hidden_layer_list, double bias_neuron_value){
        this.setHiddenLayerOutputValuesList(outputs_of_hidden_layer_list);
        this.setBias(bias_neuron_value);
    }

    //Methods
    public double calculateNetSum(){
        //Net sum of the inputs to the neuron multiplied by their weights
        double netSum = bias;
        for(double d: hiddenLayerOutputValuesList) {
            //For each input, multiply by it's weight in the weightsOfInputsList
            netSum = netSum + (d * weightsOfOutputsFromHiddenLayer.get(hiddenLayerOutputValuesList.indexOf(d)));
            //DEBUG
            System.out.println("[ OutputNeuron Sum function ]>>> MULTIPLYING: " + d + " and " + weightsOfOutputsFromHiddenLayer.get(hiddenLayerOutputValuesList.indexOf(d)));
            //Add to the total
        }
        System.out.println("[ HiddenNeuron net Sum function ]>>> Net sum is : " + netSum);
        //Store the value for later use
        setNetSumOfTheInputsToTheOutputNeuron(netSum);

        return netSum;
    }

    public double calculateSigmoidOutput(double netInput){
        //Sigmoid activation function
        double outputValue = (1/(1 + Math.exp(-netInput)));
        //DEBUG
        System.out.println("[ OutputNeuron Sigmoid function ]>>> Output of sigmoid: " + outputValue);
        //Store the value for later use
        setSigmoidOutputOfTheOutputNeuron(outputValue);
        return outputValue;
    }

    //Fires the neuron and produces an output
    public double fire(){
        double sigmoidOutput = calculateSigmoidOutput(calculateNetSum());
        return sigmoidOutput;
    }

    //Get and Set
    public double getTargetOutput() {
        return targetOutput;
    }

    public void setTargetOutput(double targetOutput) {
        this.targetOutput = targetOutput;
    }

    public double getActualOutput() {
        return actualOutput;
    }

    public void setActualOutput(double actualOutput) {
        this.actualOutput = actualOutput;
    }

    public double getOutputError() {
        return outputError;
    }

    public void setOutputError(double outputError) {
        this.outputError = outputError;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public ArrayList<Double> getHiddenLayerOutputValuesList() {
        return hiddenLayerOutputValuesList;
    }

    public void setHiddenLayerOutputValuesList(ArrayList<Double> hiddenLayerOutputValuesList) {
        this.hiddenLayerOutputValuesList = hiddenLayerOutputValuesList;
    }

    public ArrayList<Double> getWeightsOfOutputsFromHiddenLayer() {
        return weightsOfOutputsFromHiddenLayer;
    }

    public void setWeightsOfOutputsFromHiddenLayer(ArrayList<Double> weightsOfOutputsFromHiddenLayer) {
        this.weightsOfOutputsFromHiddenLayer = weightsOfOutputsFromHiddenLayer;
    }

    public double getSigmoidOutputOfTheOutputNeuron() {
        return sigmoidOutputOfTheOutputNeuron;
    }

    public void setSigmoidOutputOfTheOutputNeuron(double sigmoidOutputOfTheOutputNeuron) {
        this.sigmoidOutputOfTheOutputNeuron = sigmoidOutputOfTheOutputNeuron;
    }

    public double getNetSumOfTheInputsToTheOutputNeuron() {
        return netSumOfTheInputsToTheOutputNeuron;
    }

    public void setNetSumOfTheInputsToTheOutputNeuron(double netSumOfTheInputsToTheOutputNeuron) {
        this.netSumOfTheInputsToTheOutputNeuron = netSumOfTheInputsToTheOutputNeuron;
    }
}
