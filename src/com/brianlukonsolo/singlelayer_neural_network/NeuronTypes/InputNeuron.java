package com.brianlukonsolo.singlelayer_neural_network.NeuronTypes;

import java.util.ArrayList;

/**
 * Created by lordmcbrian on 29/05/2017.
 */
public class InputNeuron {
    //Each input neuron only has forward connections in the network, from left to right
    // InputNeuron --> [[[O]]] ------ HiddenNeuron [o] -------- OutputNeuron [o]

    //Each input-neuron has one input
    private double inputValue;
    //Connections to hidden layer
    private ArrayList<Double> hiddenLayerWeights = new ArrayList<>();

    public InputNeuron(double inputValue){
        this.setInputValue(inputValue);
    }

    //Get and Set
    public double getInputValue() {
        return inputValue;
    }

    public void setInputValue(double inputValue) {
        this.inputValue = inputValue;
    }
}
