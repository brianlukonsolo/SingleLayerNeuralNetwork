package com.brianlukonsolo.singlelayer_neural_network.Network;

import com.brianlukonsolo.singlelayer_neural_network.NeuronTypes.HiddenNeuron;
import com.brianlukonsolo.singlelayer_neural_network.NeuronTypes.InputNeuron;
import com.brianlukonsolo.singlelayer_neural_network.NeuronTypes.OutputNeuron;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/**
 * Created by lordmcbrian on 29/05/2017.
 */
public class Network {
    //Initialise default learning-rate
    private double learningRate = 0.5;
    private double biasNeuronValue = 0.35;
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
    public Network(ArrayList<Double> inputsForTheHiddenLayer) {
        //store the inputs
        this.inputsList = inputsForTheHiddenLayer;
        //Run the network once with the inputs to initialise all the neurons, Weights and Input Arrays
        this.configureNeuralNetwork(inputsForTheHiddenLayer);
    }

    public Network(ArrayList<InputNeuron> input_neurons_list, ArrayList<HiddenNeuron> hidden_neurons_list, ArrayList<OutputNeuron> output_neuron_list) {
        this.setInputNeuronsList(input_neurons_list);
        this.setHiddenNeuronsList(hidden_neurons_list);
        this.setOutputNeuronsList(output_neuron_list);
    }

    //Methods
    public ArrayList<Double> startLearning() {
        //Configure
        configureNeuralNetwork(getInputsList());
        //Foward-propagate
        setActualOutputsList(fowardPropagate());

        //Arraylist to hold the outputs of the network after the forward pass and backpropagation
        ArrayList<Double> networkOutputs = new ArrayList<>();

        //TODO:implement foward propagation and backpropagation here

        return networkOutputs;
    }

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
        input_layer.add(new InputNeuron(0.05));
        input_layer.add(new InputNeuron(0.10));
        //TODO: UNCOMMENT FOR LOOP AND DELETE THE ABOVE NEURONS
       /* ArrayList<Double> inputValuesListForNetwork = new ArrayList<>();
        for(double d: inputs_for_the_network){
            inputValuesListForNetwork.add(d);
        } */

        //
        ArrayList<Double> weightsOfInputs = allWeightsBetweenInputAndHiddenLayer;
        //TODO: UNCOMMENT FOR LOOP AND DELETE ABOVE WEIGHT LIST
        //Create arraylist holding random weights for the inputs
        /*ArrayList<Double> weightsOfInputs = new ArrayList<>();
        for(InputNeuron inp: input_layer){
            weightsOfInputs.add(random.nextDouble());
        }*/

        //Create the input layer
        setInputNeuronsList(input_layer);
        //------------------------------------------------------------------
        System.out.println(" END OF INPUT LAYER CREATION ");


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
        setHiddenNeuronsList(hidden_layer);

        //===============================================================================================================
        //----------------------- HIDDEN NEURON LOOP TO CALCULATE NET SUM AND OUTPUT
        //===============================================================================================================
        int shift_index = 0;
        for(int h = 0; h < getHiddenNeuronsList().size() ; h++ ){
            System.out.println("===========================================================>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> HIDDEN NEURON");
            double w1 = allWeightsBetweenInputAndHiddenLayer.get(h + shift_index);
            double w2 = allWeightsBetweenInputAndHiddenLayer.get(h + shift_index + 1);
            //Add the relevant weights for the neuron
            ArrayList<Double> relevantWeights = new ArrayList<>();
            relevantWeights.add(w1);
            relevantWeights.add(w2);
            //Increment the shift index
            shift_index = shift_index + (getInputsList().size()-1);
            System.out.println(" Weights needed for net sum  >>>> " + relevantWeights);

            //Calculate the net sum and output values of the hidden neuron
            double output = getHiddenNeuronsList().get(h).fire(relevantWeights);
        }


        //------------------------------------------------------------------
        System.out.println(" END OF HIDDEN LAYER CREATION ");

        //Create output layer
        ArrayList<OutputNeuron> output_layer = new ArrayList<>();
        //Add output neurons (as many as specified in NUMBER_OF_OUTPUT_LAYER_NEURONS)
        //Outputs of the hidden layer are inputs to the output layer so we must activate their calculations at least once
        ArrayList<Double> inputsForOutputNeurons = new ArrayList<>();
        for (HiddenNeuron hiddenNeuron : hidden_layer) {
            inputsForOutputNeurons.add(hiddenNeuron.getSigmoidOutputOfTheHiddenNeuron());
        }

        //add two output neurons to the output layer----------------------------------------OUTPUT NEURON LAYER
        OutputNeuron out1 = new OutputNeuron(inputsForOutputNeurons, 0.60, 0.01);
        out1.setWeightsOfOutputsFromHiddenLayer(allWeightsBetweenHiddenAndOutputLayer);
        output_layer.add(out1);

        OutputNeuron out2 = new OutputNeuron(inputsForOutputNeurons, 0.60, 0.99);
        out2.setWeightsOfOutputsFromHiddenLayer(allWeightsBetweenHiddenAndOutputLayer);
        output_layer.add(out2);

        //Set output layer
        setOutputNeuronsList(output_layer);

        //Finally, set the target-outputs and actual outputs list
        //It contains a list of target outputs, each coressponding to an output neuron
        for (OutputNeuron o : getOutputNeuronsList()) {
            targetOutputsList.add(o.getTargetOutput());
        }

        //===============================================================================================================
        //----------------------- OUTPUT NEURON LOOP TO CALCULATE NET SUM AND OUTPUT AND ERRORS
        //===============================================================================================================
        int shift_index_o = 0;
        for(int x = 0; x < getHiddenNeuronsList().size() ; x++ ){
            System.out.println("===========================================================<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< OUTPUT NEURON");
            double w1 = allWeightsBetweenHiddenAndOutputLayer.get(x + shift_index_o);
            double w2 = allWeightsBetweenHiddenAndOutputLayer.get(x + shift_index_o + 1);
            //Add the relevant weights for the neuron
            ArrayList<Double> relevantWeights_o = new ArrayList<>();
            relevantWeights_o.add(w1);
            relevantWeights_o.add(w2);
            //Increment the shift index for output neurons
            shift_index_o = shift_index_o + (getInputsList().size()-1);

            System.out.println(" Weights needed for net sum  >>>> " + relevantWeights_o);

            //Calculate the net sum and output values of the hidden neuron
            double output = getOutputNeuronsList().get(x).fire(relevantWeights_o);
            actualOutputsList.add(output);
        }

        System.out.println(" END OF OUTPUT LAYER CREATION ");

        //===============================================================================================================
        //-----------------------BACKPROPAGATION LOOP
        //===============================================================================================================
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

            System.out.println(" :::::::::::::::::::::::::::::::::::::::::::::::::::::: BACKPROPAGATION ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: ++++++");
            for (w = (w + (shifter)); w < weightsOfInputs.size(); w++) {
                System.out.println("===========Now processing hidden neuron " + w);

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
                System.out.println(">>>> INPUT 1 --> " + input_1);
                System.out.println(">>>> INPUT 2 --> " + input_2);

                //We need the outputs of the hidden neurons during the calculation of the partial derivatives
                //The output will change as the for loop advances
                double output_of_hidden_neuron = getHiddenNeuronsList().get(w).getSigmoidOutputOfTheHiddenNeuron();

                //Calculating the weights: w1, w2, w3, and w4.
                //These are located between the input layer and the hidden layer
                double outputError01 = (-(getOutputNeuronsList().get(0).getTargetOutput() - getOutputNeuronsList().get(0).getActualOutput())) * ( getOutputNeuronsList().get(0).getActualOutput() * (1 - getOutputNeuronsList().get(0).getActualOutput())) * weight_1_right;
                double outputError02 = (-(getOutputNeuronsList().get(1).getTargetOutput() - getOutputNeuronsList().get(1).getActualOutput())) * ( getOutputNeuronsList().get(1).getActualOutput() * (1 - getOutputNeuronsList().get(1).getActualOutput())) * weight_2_right;
                System.out.println(">>> OUTPUT ERROR 1: " + outputError01);
                System.out.println(">>> OUTPUT ERROR 2: " + outputError02);
                double totalOutputError = outputError01 + outputError02;
                //This uses input 1
                double error_for_equation_1 = (totalOutputError * (getHiddenNeuronsList().get(hiddenNeuronIndex).getSigmoidOutputOfTheHiddenNeuron() * (1 - getHiddenNeuronsList().get(hiddenNeuronIndex).getSigmoidOutputOfTheHiddenNeuron())) * input_1);
                //This uses input 2
                double error_for_equation_2 = (totalOutputError * (getHiddenNeuronsList().get(hiddenNeuronIndex).getSigmoidOutputOfTheHiddenNeuron() * (1 - getHiddenNeuronsList().get(hiddenNeuronIndex).getSigmoidOutputOfTheHiddenNeuron())) * input_2);
                //Increment the hidden neuron index so that the next pair of calculations can use the correct hidden neuron output for the differentiation calculation
                hiddenNeuronIndex = hiddenNeuronIndex + 1;
                System.out.println("HIDDEN NEURON INDEX IS -========>>>>>> " + hiddenNeuronIndex);
                System.out.println("ERROR FOR EQUATION 1 ==> " + error_for_equation_1);
                System.out.println("ERROR FOR EQUATION 2 ==> " + error_for_equation_2);
                double updatedWeight_1 = (weight_1_left - (learningRate * error_for_equation_1));
                double updatedWeight_2 = (weight_2_left - (learningRate * error_for_equation_2));
                System.out.println("THE UPDATED WEIGHTS: " + updatedWeight_1 + ", " + updatedWeight_2);
                //Add the updated weights to the updated weights list
                updated_weights_between_inputs_and_hidden_layers.add(updatedWeight_1);
                updated_weights_between_inputs_and_hidden_layers.add(updatedWeight_2);

                //Calculating the weights: w5, w6, w7, and w8.
                //These are located between the hidden layer and the output layer
                double weight_error_1 = (-(getOutputNeuronsList().get(outputNeuronIndex).getTargetOutput() - getOutputNeuronsList().get(outputNeuronIndex).getActualOutput())) * ( getOutputNeuronsList().get(outputNeuronIndex).getActualOutput() * (1 - getOutputNeuronsList().get(outputNeuronIndex).getActualOutput())) * getHiddenNeuronsList().get(0).getSigmoidOutputOfTheHiddenNeuron();
                double weight_error_2 = (-(getOutputNeuronsList().get(outputNeuronIndex).getTargetOutput() - getOutputNeuronsList().get(outputNeuronIndex).getActualOutput())) * ( getOutputNeuronsList().get(outputNeuronIndex).getActualOutput() * (1 - getOutputNeuronsList().get(outputNeuronIndex).getActualOutput())) * getHiddenNeuronsList().get(1).getSigmoidOutputOfTheHiddenNeuron();
                //Increment output neuron index
                outputNeuronIndex = outputNeuronIndex + 1;
                double updatedWeight_1_right = weight_1_right - (learningRate * weight_error_1);
                double updatedWeight_2_right = weight_2_right - (learningRate * weight_error_2);
                updated_weights_between_hidden_layer_and_output_layer.add(updatedWeight_1_right);
                updated_weights_between_hidden_layer_and_output_layer.add(updatedWeight_2_right);


                //PRINT THE VALUES
                System.out.println("    # LEFT SIDE WEIGHTS ARE > " + weight_1_left + ", " + weight_2_left);
                System.out.println("    # RIGHT SIDE WEIGHTS ARE > " + weight_1_right + ", " + weight_2_right);
                System.out.println("INPUT VALUES AT THIS POINT = " + input_1 + ", " + input_2); //Need inputs 1 and 2
                System.out.println("    #> Output of hidden neuron = " + output_of_hidden_neuron);
                //TODO: SIGMOID OUTPUT OF HIDDEN NEURONS IS DUPLICATED!!! FIX THIS!!! loop through the hidden neurons, calculating correct sigmoid outputs for each!
                if (shifter + 1 < getInputsList().size()) {
                    shifter = shifter + 1;
                }
                inputIndex = inputIndex + 1;
                if (inputIndex >= getInputsList().size()) {
                    inputIndex = 0;
                }
                //System.out.println("INPUTS LIST SIZE: " + inputsList.size());

                //If w increases to more than the number of inputs, end the loop
                if (w >= getInputsList().size() - 1) {
                    break;
                }

                //TODO: NEED TO STORE THE UPDATED-WEIGHTS UNTIL ITS TIME TO ACTUALLY UPDATE THE WEIGHTS
            }
            //Increase index of where to begin getting weights in the list for each hidden neuron

            System.out.println("#### FINALLY ---> The Updated weights between input layer and hidden layer: " + updated_weights_between_inputs_and_hidden_layers);
        System.out.println("#### FINALLY ---> The Updated weights between hidden layer and output layer: " + updated_weights_between_hidden_layer_and_output_layer);

    }

    //The foward pass of the neural network
    public ArrayList<Double> fowardPropagate() {
        ArrayList<Double> inputs_for_the_network = getInputsList();
        //TODO: IMPLEMENT FORWARD PROPAGATION!!!!!
        ArrayList<Double> outputs = new ArrayList<>();

        //TODO: IMPLEMENT LOGIC

        return outputs;
    }

    //TODO: UNDER CONSTRUCTION===============================================================================================####
    //An implementation of the Backpropagation algorithm based on the calculation of partial derivatives
    public void backpropagate() {
        //TODO: Implement backpropagation algorithm for the network
        //Storage for the updated weights after backpropagation
        ArrayList<Double> updatedWeights_InputsToHiddenLayer = new ArrayList();
        ArrayList<Double> updatedWeights_HiddenLayerToOutputs = new ArrayList();

        //TODO: IMPLEMENT LOGIC

        //Finally update ALL the weights in the network
        //TODO: Finally, update all the weights in the network!! (dont forget to give the neurons the updated weights if still giving them this responsibility of knowing the full list)
    }

    public void updateNetworkWeights(){

        //TODO: IMPLEMENT
    }
    //TODO: END OF - UNDER CONSTRUCTION===============================================================================================####

    //This method calculates the individual output error of each neuron and then sums all of them to get the total output error
    public double calculateOutputErrorForEachOutputNeuron(ArrayList<OutputNeuron> output_neurons_list) {
        //clear the error list and update with new values
        errorList.clear();

        double totalOutputError = 0;

        for (OutputNeuron outputNeuron : output_neurons_list) {
            //Reminder: This outputs correctly. Do not modify.
            double target = getTargetOutputsList().get(output_neurons_list.indexOf(outputNeuron));
            double output = outputNeuron.getActualOutput();
            System.out.println(">>> T - O = " + (target - output));
            //Squared error equation is = 1/2(target - output^2)
            //We use 1.0/2.0 to represent 1/2 because we are dealing with doubles
            double error = ((1.0 / 2.0) * Math.pow((target - output), 2));
            System.out.println("NEURON ERROR = " + error);
            //Let the neuron know its error
            outputNeuron.setOutputError(error);
            //Add to the total error
            totalOutputError = totalOutputError + error;
        }
        System.out.println("Total output error is : " + totalOutputError);
        return totalOutputError;
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
}
