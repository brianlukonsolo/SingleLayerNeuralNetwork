package com.brianlukonsolo.singlelayer_neural_network.Network;

import com.brianlukonsolo.singlelayer_neural_network.NeuronTypes.HiddenNeuron;
import com.brianlukonsolo.singlelayer_neural_network.NeuronTypes.InputNeuron;
import com.brianlukonsolo.singlelayer_neural_network.NeuronTypes.OutputNeuron;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/**
 * Created by Brian Lukonsolo on 29/05/2017.
 */

//Quick note on orientation (when visualising the network)
//Arrangement of the inputs and weights: TOP-TO-BOTTOM, with input1 on the top left and the last input in the bottom left corner.
// This is assuming you have drawn a neural network on a piece of paper and the network runs its forward pass from
// left to right (ie: inputs to the left, outputs to the right)

public class Network {
    //Initialise default learning-rate
    private double learningRate = 0.5;
    private double biasNeuronValue = 0.35;
    private double acceptableDeviation = 0.05;
    private double epochLimit = 1500;
    private int NUMBER_OF_HIDDEN_LAYER_NEURONS = 2; //TODO: Set this somehow in the main function
    private int NUMBER_OF_OUTPUT_LAYER_NEURONS = 2;

    //These are storage
    private ArrayList<Double> inputsList = new ArrayList<>();
    //The network will aim to achieve these targets
    private ArrayList<Double> targetOutputsList = new ArrayList<>(); //TODO: implement targets in the code
    //The actual results of the forward pass algorithm
    private ArrayList<Double> actualOutputsList = new ArrayList<>();
    private ArrayList<Double> errorList = new ArrayList<>(); //TODO: implement error list in the code

    //A network contains input neurons, hidden neurons and output neurons
    private ArrayList<InputNeuron> inputNeuronsList = new ArrayList<>();
    private ArrayList<HiddenNeuron> hiddenNeuronsList = new ArrayList<>();
    private ArrayList<OutputNeuron> outputNeuronsList = new ArrayList<>();

    //Constructors
    public Network(ArrayList<Double> inputsForTheHiddenLayer, ArrayList<Double> targetOutputsList) {
        //store the inputs
        this.inputsList = inputsForTheHiddenLayer;
        this.targetOutputsList = targetOutputsList;
        //Run the network once with the inputs to initialise all the neurons, Weights and Input Arrays
        this.configureNeuralNetwork(inputsForTheHiddenLayer);
    }

    public Network(ArrayList<InputNeuron> input_neurons_list, ArrayList<HiddenNeuron> hidden_neurons_list, ArrayList<OutputNeuron> output_neuron_list) {
        this.setInputNeuronsList(input_neurons_list);
        this.setHiddenNeuronsList(hidden_neurons_list);
        this.setOutputNeuronsList(output_neuron_list);
    }

    //Methods
    public void configureNeuralNetwork(ArrayList<Double> inputs_for_the_network) {
        Random random = new Random();
        random.setSeed(System.currentTimeMillis());

        //TEMPORARY LISTS OF WEIGHTS:
        ArrayList<Double> allWeightsBetweenInputAndHiddenLayer = new ArrayList<>(Arrays.asList(0.15, 0.20, 0.25, 0.30));
        ArrayList<Double> allWeightsBetweenHiddenAndOutputLayer = new ArrayList<>(Arrays.asList(0.4, 0.45, 0.50, 0.55));

        //Create input layer neurons
        //For each input, create an input object in the input layer
        ArrayList<InputNeuron> input_layer = new ArrayList<>();
        //TODO: USE FOR LOOP
        input_layer.add(new InputNeuron(getInputsList().get(0)));
        input_layer.add(new InputNeuron(getInputsList().get(1)));
        setInputNeuronsList(input_layer);
        //END OF INPUT LAYER CREATION


        //Create the hidden layer neurons
        ArrayList<HiddenNeuron> hidden_layer = new ArrayList<>();
        //hidden neuron 1
        HiddenNeuron h1 = new HiddenNeuron(inputs_for_the_network, 0.35);
        h1.setWeightsOfInputs(allWeightsBetweenInputAndHiddenLayer);
        h1.setInputsLayerValuesList(getInputsList());
        hidden_layer.add(h1);

        //hidden neuron 1
        HiddenNeuron h2 = new HiddenNeuron(inputs_for_the_network, 0.35);
        h2.setWeightsOfInputs(allWeightsBetweenInputAndHiddenLayer);
        h2.setInputsLayerValuesList(getInputsList());
        hidden_layer.add(h2);

        //Create the hidden layer
        //
        /////////////////////////
        setHiddenNeuronsList(hidden_layer);

        //===============================================================================================================
        //----------------------- HIDDEN NEURON LOOP TO CALCULATE NET SUM AND OUTPUT
        //===============================================================================================================
        int shift_index = 0;
        for (int h = 0; h < getHiddenNeuronsList().size(); h++) {
            double w1 = allWeightsBetweenInputAndHiddenLayer.get(h + shift_index);
            double w2 = allWeightsBetweenInputAndHiddenLayer.get(h + shift_index + 1);
            //Add the relevant weights for the neuron
            ArrayList<Double> relevantWeights = new ArrayList<>();
            relevantWeights.add(w1);
            relevantWeights.add(w2);
            //Increment the shift index
            shift_index = shift_index + (getInputsList().size() - 1);

            //Calculate the net sum and output values of the hidden neuron
            double output = getHiddenNeuronsList().get(h).fire(relevantWeights);
        }


        //Create output layer
        //
        /////////////////////
        ArrayList<OutputNeuron> output_layer = new ArrayList<>();
        //Add output neurons (as many as specified in NUMBER_OF_OUTPUT_LAYER_NEURONS)
        //Outputs of the hidden layer are inputs to the output layer so we must activate their calculations at least once
        ArrayList<Double> inputsForOutputNeurons = new ArrayList<>();
        for (HiddenNeuron hiddenNeuron : hidden_layer) {
            inputsForOutputNeurons.add(hiddenNeuron.getSigmoidOutputOfTheHiddenNeuron());
        }
        //add two output neurons to the output layer----------------------------------------OUTPUT NEURON LAYER
        OutputNeuron out1 = new OutputNeuron(inputsForOutputNeurons, 0.60, getTargetOutputsList().get(0));
        out1.setWeightsOfOutputsFromHiddenLayer(allWeightsBetweenHiddenAndOutputLayer);
        output_layer.add(out1);

        OutputNeuron out2 = new OutputNeuron(inputsForOutputNeurons, 0.60, getTargetOutputsList().get(1));
        out2.setWeightsOfOutputsFromHiddenLayer(allWeightsBetweenHiddenAndOutputLayer);
        output_layer.add(out2);

        //Set output layer
        setOutputNeuronsList(output_layer);

        //Finally, set the target-outputs and actual outputs list
        //It contains a list of target outputs, each corresponding to an output neuron
        for (OutputNeuron o : getOutputNeuronsList()) {
            targetOutputsList.add(o.getTargetOutput());
        }
        //===============================================================================================================
        //----------------------- OUTPUT NEURON LOOP TO CALCULATE NET SUM AND OUTPUT AND ERRORS
        //===============================================================================================================
        int shift_index_o = 0;
        for (int x = 0; x < getHiddenNeuronsList().size(); x++) {
            double w1 = allWeightsBetweenHiddenAndOutputLayer.get(x + shift_index_o);
            double w2 = allWeightsBetweenHiddenAndOutputLayer.get(x + shift_index_o + 1);
            //Add the relevant weights for the neuron
            ArrayList<Double> relevantWeights_o = new ArrayList<>();
            relevantWeights_o.add(w1);
            relevantWeights_o.add(w2);
            //Increment the shift index for output neurons
            shift_index_o = shift_index_o + (getInputsList().size() - 1);

            //Calculate the net sum and output values of the hidden neuron
            double output = getOutputNeuronsList().get(x).fire(relevantWeights_o);
            actualOutputsList.add(output);
        }
        //END OF OUTPUT LAYER CREATION
    }

    //The forward pass of the neural network
    public void fowardPropagate() {
        ArrayList<Double> allWeightsBetweenInputAndHiddenLayer = getHiddenNeuronsList().get(0).getWeightsOfInputs();
        ArrayList<Double> allWeightsBetweenHiddenAndOutputLayer = getOutputNeuronsList().get(0).getWeightsOfOutputsFromHiddenLayer();
        //Storage for the outputs from the output neurons
        ArrayList<Double> outputs = new ArrayList<>();
        //===============================================================================================================
        //----------------------- HIDDEN NEURON LOOP TO CALCULATE NET SUM AND OUTPUT
        //===============================================================================================================
        int shift_index = 0;
        for (int h = 0; h < getHiddenNeuronsList().size(); h++) {
            double w1 = allWeightsBetweenInputAndHiddenLayer.get(h + shift_index);
            double w2 = allWeightsBetweenInputAndHiddenLayer.get(h + shift_index + 1);
            //Add the relevant weights for the neuron
            ArrayList<Double> relevantWeights = new ArrayList<>();
            relevantWeights.add(w1);
            relevantWeights.add(w2);
            //Increment the shift index
            shift_index = shift_index + (getInputsList().size() - 1);

            //Calculate the net sum and output values of the hidden neuron
            double output = getHiddenNeuronsList().get(h).fire(relevantWeights);
        }

        //===============================================================================================================
        //----------------------- OUTPUT NEURON LOOP TO CALCULATE NET SUM AND OUTPUT AND ERRORS
        //===============================================================================================================
        //Clear the actual outputs list
        actualOutputsList.clear();

        int shift_index_o = 0;
        for (int x = 0; x < getHiddenNeuronsList().size(); x++) {
            double w1 = allWeightsBetweenHiddenAndOutputLayer.get(x + shift_index_o);
            double w2 = allWeightsBetweenHiddenAndOutputLayer.get(x + shift_index_o + 1);
            //Add the relevant weights for the neuron
            ArrayList<Double> relevantWeights_o = new ArrayList<>();
            relevantWeights_o.add(w1);
            relevantWeights_o.add(w2);
            //Increment the shift index for output neurons
            shift_index_o = shift_index_o + (getInputsList().size() - 1);

            //Calculate the net sum and output values of the hidden neuron
            double output = getOutputNeuronsList().get(x).fire(relevantWeights_o);
            actualOutputsList.add(output);
        }

    }

    public void backpropagate() {
        ArrayList<Double> allWeightsBetweenInputAndHiddenLayer = getHiddenNeuronsList().get(0).getWeightsOfInputs();
        ArrayList<Double> allWeightsBetweenHiddenAndOutputLayer = getOutputNeuronsList().get(0).getWeightsOfOutputsFromHiddenLayer();
        ArrayList<Double> weightsOfInputs = allWeightsBetweenInputAndHiddenLayer;

        //FOR EACH HIDDEN NEURON IN THE HIDDEN LAYER WE WANT TO CALCULATE THE CORRECT NET SUM
        int shifter = 0;
        int w = 0;
        int right_side_index = 0;
        int inputIndex = 0;
        int index_static = 0;
        int hiddenNeuronIndex = 0;
        int outputNeuronIndex = 0;
        //Storage for the updated weights
        ArrayList<Double> updated_weights_between_inputs_and_hidden_layers = new ArrayList<>();
        ArrayList<Double> updated_weights_between_hidden_layer_and_output_layer = new ArrayList<>();

        //:::::::::::::::::::::::::::::::::::::::::::::::::::::: BACKPROPAGATION :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        for (w = (w + (shifter)); w < weightsOfInputs.size(); w++) {
            //The required weights between the input layer and the hidden layer
            double weight_1_left = weightsOfInputs.get(w + shifter);
            double weight_2_left = weightsOfInputs.get(w + 1 + shifter);
            //The required weights between the hidden layer and the output layer
            double weight_1_right = allWeightsBetweenHiddenAndOutputLayer.get(w + right_side_index);
            double weight_2_right = allWeightsBetweenHiddenAndOutputLayer.get(w + right_side_index + 1);
            right_side_index = right_side_index + 1;

            //Note: The first calculation uses the first input. The second calculation uses the second input.
            double input_1 = getInputsList().get(index_static);
            double input_2 = getInputsList().get(index_static + 1);

            //We need the outputs of the hidden neurons during the calculation of the partial derivatives
            //The output will change as the for loop advances
            double output_of_hidden_neuron = getHiddenNeuronsList().get(w).getSigmoidOutputOfTheHiddenNeuron();

            //Calculating the weights: w1, w2, w3, and w4.
            //These are located between the input layer and the hidden layer
            double outputError01 = (-(getOutputNeuronsList().get(0).getTargetOutput() - getOutputNeuronsList().get(0).getActualOutput())) * (getOutputNeuronsList().get(0).getActualOutput() * (1 - getOutputNeuronsList().get(0).getActualOutput())) * weight_1_right;
            double outputError02 = (-(getOutputNeuronsList().get(1).getTargetOutput() - getOutputNeuronsList().get(1).getActualOutput())) * (getOutputNeuronsList().get(1).getActualOutput() * (1 - getOutputNeuronsList().get(1).getActualOutput())) * weight_2_right;
            double totalOutputError = outputError01 + outputError02;
            //This uses input 1
            double error_for_equation_1 = (totalOutputError * (getHiddenNeuronsList().get(hiddenNeuronIndex).getSigmoidOutputOfTheHiddenNeuron() * (1 - getHiddenNeuronsList().get(hiddenNeuronIndex).getSigmoidOutputOfTheHiddenNeuron())) * input_1);
            //This uses input 2
            double error_for_equation_2 = (totalOutputError * (getHiddenNeuronsList().get(hiddenNeuronIndex).getSigmoidOutputOfTheHiddenNeuron() * (1 - getHiddenNeuronsList().get(hiddenNeuronIndex).getSigmoidOutputOfTheHiddenNeuron())) * input_2);
            //Increment the hidden neuron index so that the next pair of calculations can use the correct hidden neuron output for the differentiation calculation
            hiddenNeuronIndex = hiddenNeuronIndex + 1;

            double updatedWeight_1 = (weight_1_left - (learningRate * error_for_equation_1));
            double updatedWeight_2 = (weight_2_left - (learningRate * error_for_equation_2));
            //Add the updated weights to the updated weights list
            updated_weights_between_inputs_and_hidden_layers.add(updatedWeight_1);
            updated_weights_between_inputs_and_hidden_layers.add(updatedWeight_2);

            //Calculating the weights: w5, w6, w7, and w8.
            //These are located between the hidden layer and the output layer
            double weight_error_1 = (-(getOutputNeuronsList().get(outputNeuronIndex).getTargetOutput() - getOutputNeuronsList().get(outputNeuronIndex).getActualOutput())) * (getOutputNeuronsList().get(outputNeuronIndex).getActualOutput() * (1 - getOutputNeuronsList().get(outputNeuronIndex).getActualOutput())) * getHiddenNeuronsList().get(0).getSigmoidOutputOfTheHiddenNeuron();
            double weight_error_2 = (-(getOutputNeuronsList().get(outputNeuronIndex).getTargetOutput() - getOutputNeuronsList().get(outputNeuronIndex).getActualOutput())) * (getOutputNeuronsList().get(outputNeuronIndex).getActualOutput() * (1 - getOutputNeuronsList().get(outputNeuronIndex).getActualOutput())) * getHiddenNeuronsList().get(1).getSigmoidOutputOfTheHiddenNeuron();
            //Increment output neuron index
            outputNeuronIndex = outputNeuronIndex + 1;
            double updatedWeight_1_right = weight_1_right - (learningRate * weight_error_1);
            double updatedWeight_2_right = weight_2_right - (learningRate * weight_error_2);
            updated_weights_between_hidden_layer_and_output_layer.add(updatedWeight_1_right);
            updated_weights_between_hidden_layer_and_output_layer.add(updatedWeight_2_right);
            //Increment shifter
            if (shifter + 1 < getInputsList().size()) {
                shifter = shifter + 1;
            }
            inputIndex = inputIndex + 1;
            if (inputIndex >= getInputsList().size()) {
                inputIndex = 0;
            }
            //If w increases to more than the number of inputs, end the loop
            if (w >= getInputsList().size() - 1) {
                break;
            }
        }

        //===============================================================================================================
        //-----------------------UPDATE NETWORK WEIGHTS
        //===============================================================================================================
        for (int i = 0; i < getHiddenNeuronsList().size(); i++) {
            getHiddenNeuronsList().get(i).setWeightsOfInputs(updated_weights_between_inputs_and_hidden_layers);
        }
        for (int i = 0; i < getOutputNeuronsList().size(); i++) {
            getOutputNeuronsList().get(i).setWeightsOfOutputsFromHiddenLayer(updated_weights_between_hidden_layer_and_output_layer);
        }

        System.out.println("#### Updated weights between input layer and hidden layer: " + updated_weights_between_inputs_and_hidden_layers);
        System.out.println("#### Updated weights between hidden layer and output layer: " + updated_weights_between_hidden_layer_and_output_layer);
    }

    public void learn() {
        //Target and actual outputs
        double target_1 = getTargetOutputsList().get(0);
        double target_2 = getTargetOutputsList().get(1);
        double actual_1;
        double actual_2;
        //Keep track of epochs
        int epochCount = 0;
        //Forward propagate and backpropagate until the output error is minimal
        do {
            System.out.println("\n#######====== EPOCH " + epochCount + " =====#######\n");
            //Get the current Actual Outputs
            actual_1 = getActualOutputsList().get(0);
            actual_2 = getActualOutputsList().get(1);
            //Foward propagate
            fowardPropagate();
            //Backpropagate
            backpropagate();
            System.out.println("[ Backpropagation ]=> Network outputs : " + actualOutputsList);
            epochCount = epochCount + 1;
            System.out.println("#############################\n");
            //Check if the network has reached its target and if so, stop
            if (epochCount >= epochLimit) {
                System.out.println("\n#######====== EPOCH " + epochCount + " =====#######\n");
                System.out.println("EPOCH LIMIT REACHED!!!");
                System.out.println("Final outputs were: " + actualOutputsList);
                break;
            }

            if ((actual_1 < target_1 + getAcceptableDeviation() && actual_1 > target_1 - getAcceptableDeviation()) && (actual_2 < target_2 + getAcceptableDeviation() && actual_2 > target_2 - getAcceptableDeviation())) {
                System.out.println("[[[ ------------------------------------------------------------------------------------------------------------------------------------------ ]]]");
                System.out.println("[[[ ===== NEURAL NETWORK SUCCESSFULLY TRAINED! ===== ]]]------------------------------------------------------------------------------[ SUCCESS! ]");
                System.out.println("[[[ ------------------------------------------------------------------------------------------------------------------------------------------ ]]]");
                System.out.println("Training complete in: " + epochCount + " epochs.\n");
                System.out.println("Final outputs: " + actualOutputsList);
                System.out.println("\n---- Data ----\nIteration limit was set to: " + getEpochLimit());
                System.out.println("Learning rate was set to: " + getLearningRate());
                System.out.println("Inputs were: " + getInputsList());
                System.out.println("Acceptable deviation was set to: " + getAcceptableDeviation());
                System.out.println("Hidden layer bias was set to: " + getHiddenNeuronsList().get(0).getBias());
                System.out.println("Output layer bias was set to: " + getOutputNeuronsList().get(0).getBias());
                System.out.println("[[[ ===== -------------------------------------------------------------------------------------------------------------------[ TRAINING COMPLETE ]");
                System.out.println("[[[ ------------------------------------------------------------------------------------------------------------------------------------------ ]]]");
                break;
            }
        } while (epochCount <= epochLimit + 1);
    }

    //Get and Set
    public ArrayList<InputNeuron> getInputNeuronsList() {
        return inputNeuronsList;
    }

    public void setInputNeuronsList(ArrayList<InputNeuron> inputNeuronsList) {
        this.inputNeuronsList = inputNeuronsList;
    }

    public ArrayList<HiddenNeuron> getHiddenNeuronsList() {
        return hiddenNeuronsList;
    }

    public void setHiddenNeuronsList(ArrayList<HiddenNeuron> hiddenNeuronsList) {
        this.hiddenNeuronsList = hiddenNeuronsList;
    }

    public ArrayList<OutputNeuron> getOutputNeuronsList() {
        return outputNeuronsList;
    }

    public void setOutputNeuronsList(ArrayList<OutputNeuron> outputNeuronsList) {
        this.outputNeuronsList = outputNeuronsList;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public ArrayList<Double> getTargetOutputsList() {
        return targetOutputsList;
    }

    public void setTargetOutputsList(ArrayList<Double> targetOutputsList) {
        this.targetOutputsList = targetOutputsList;
    }

    public ArrayList<Double> getActualOutputsList() {
        return actualOutputsList;
    }

    public void setActualOutputsList(ArrayList<Double> actualOutputsList) {
        this.actualOutputsList = actualOutputsList;
    }

    public ArrayList<Double> getInputsList() {
        return inputsList;
    }

    public void setInputsList(ArrayList<Double> inputs) {
        this.inputsList = inputs;
    }

    public int getNUMBER_OF_OUTPUT_LAYER_NEURONS() {
        return NUMBER_OF_OUTPUT_LAYER_NEURONS;
    }

    public void setNUMBER_OF_OUTPUT_LAYER_NEURONS(int NUMBER_OF_OUTPUT_LAYER_NEURONS) {
        this.NUMBER_OF_OUTPUT_LAYER_NEURONS = NUMBER_OF_OUTPUT_LAYER_NEURONS;
    }

    public int getNUMBER_OF_HIDDEN_LAYER_NEURONS() {
        return NUMBER_OF_HIDDEN_LAYER_NEURONS;
    }

    public void setNUMBER_OF_HIDDEN_LAYER_NEURONS(int NUMBER_OF_HIDDEN_LAYER_NEURONS) {
        this.NUMBER_OF_HIDDEN_LAYER_NEURONS = NUMBER_OF_HIDDEN_LAYER_NEURONS;
    }

    public double getBiasNeuronValue() {
        return biasNeuronValue;
    }

    public void setBiasNeuronValue(double biasNeuronValue) {
        this.biasNeuronValue = biasNeuronValue;
    }

    public ArrayList<Double> getErrorList() {
        return errorList;
    }

    public void setErrorList(ArrayList<Double> errorList) {
        this.errorList = errorList;
    }

    public double getAcceptableDeviation() {
        return acceptableDeviation;
    }

    public void setAcceptableDeviation(double acceptableDeviation) {
        this.acceptableDeviation = acceptableDeviation;
    }

    public double getEpochLimit() {
        return epochLimit;
    }

    public void setEpochLimit(double epochLimit) {
        this.epochLimit = epochLimit;
    }
}
