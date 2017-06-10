package com.brianlukonsolo.singlelayer_neural_network.NeuronTypes;

import java.util.ArrayList;

/**
 * Created by Brian Lukonsolo on 29/05/2017.
 */

//Each output neuron only has backward connections in the network, from right to left
// InputNeuron[O] ------ HiddenNeuron [o] -------- [[[o]]] <-- OutputNeuron

public class OutputNeuron {
    private ArrayList<Double> hiddenLayerOutputValuesList = new ArrayList<>();
    private ArrayList<Double> weightsOfOutputsFromHiddenLayer = new ArrayList<>();
    private double targetOutput;
    private double actualOutput;
    private double outputError;
    private double bias;
    private double sigmoidOutputOfTheOutputNeuron;
    private double netSumOfTheInputsToTheOutputNeuron;

    //Constructor
    public OutputNeuron(ArrayList<Double> outputs_of_hidden_layer_list, double bias_neuron_value, double target_output) {
        this.setHiddenLayerOutputValuesList(outputs_of_hidden_layer_list);
        this.setBias(bias_neuron_value);
        this.setTargetOutput(target_output);
    }

    public double calculateNetSum(ArrayList<Double> relevant_weights) {
        double netSum = bias;

        for (int i = 0; i < getHiddenLayerOutputValuesList().size(); i++) {
            netSum = netSum + (getHiddenLayerOutputValuesList().get(i) * relevant_weights.get(i));
        }
        setNetSumOfTheInputsToTheOutputNeuron(netSum);
        return netSum;
    }

    public double calculateSigmoidOutput(double netInput) {
        double outputValue = (1 / (1 + Math.exp(-netInput)));
        System.out.println("[ OutputNeuron Sigmoid function ]>>> Output of sigmoid: " + outputValue);
        setSigmoidOutputOfTheOutputNeuron(outputValue);
        setActualOutput(outputValue);
        return outputValue;
    }

    public double fire(ArrayList<Double> relevant_weights) {
        double sigmoidOutput = calculateSigmoidOutput(calculateNetSum(relevant_weights));
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
