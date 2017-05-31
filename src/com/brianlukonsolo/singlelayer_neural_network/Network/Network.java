package com.brianlukonsolo.singlelayer_neural_network.Network;

import com.brianlukonsolo.singlelayer_neural_network.NeuronTypes.HiddenNeuron;
import com.brianlukonsolo.singlelayer_neural_network.NeuronTypes.InputNeuron;
import com.brianlukonsolo.singlelayer_neural_network.NeuronTypes.OutputNeuron;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

/**
 * Created by lordmcbrian on 29/05/2017.
 */
public class Network {
    //Initialise default learning-rate
    private double learningRate = 0.5;
    private  double biasNeuronValue = 0.35;
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
    public Network(ArrayList<Double> inputsForTheHiddenLayer){
        //store the inputs
        this.inputsList = inputsForTheHiddenLayer;
        //Run the network once with the inputs to initialise all the neurons, Weights and Input Arrays
        this.configureNeuralNetwork(inputsForTheHiddenLayer);
    }

    public Network(ArrayList<InputNeuron > input_neurons_list, ArrayList<HiddenNeuron> hidden_neurons_list, ArrayList<OutputNeuron> output_neuron_list){
        this.setInputNeuronsList(input_neurons_list);
        this.setHiddenNeuronsList(hidden_neurons_list);
        this.setOutputNeuronsList(output_neuron_list);
    }

    //Methods
    public ArrayList<Double> startLearning(){
        //Configure
        configureNeuralNetwork(getInputsList());
        //Foward-propagate
        setActualOutputsList(fowardPropagate());

        //Arraylist to hold the outputs of the network after the forward pass and backpropagation
        ArrayList<Double> networkOutputs = new ArrayList<>();

        //TODO:implement foward propagation and backpropagation here

        return networkOutputs;
    }

    public void configureNeuralNetwork(ArrayList<Double> inputs_for_the_network){
        Random random = new Random();
        random.setSeed(System.currentTimeMillis());

        //Create input layer
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

       ArrayList<Double> weightsOfInputs = new ArrayList<>();
       weightsOfInputs.add(0.15);
       weightsOfInputs.add(0.20);
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


        //Create the hidden layer
        ArrayList<HiddenNeuron> hidden_layer = new ArrayList<>();
        //hidden neuron 1
        hidden_layer.add(new HiddenNeuron(inputs_for_the_network, 0.35));
        hidden_layer.get(0).setWeightsOfInputs(new ArrayList<Double>(Arrays.asList(0.15,0.20)));
        //Fire the neuron once or network will fail
        hidden_layer.get(0).calculateSigmoidOutput(hidden_layer.get(0).calculateNetSum());

        //hidden neuron 1
        hidden_layer.add(new HiddenNeuron(inputs_for_the_network, 0.35));
        hidden_layer.get(1).setWeightsOfInputs(new ArrayList<Double>(Arrays.asList(0.25,0.30)));
        //Fire the neuron once or network will fail
        hidden_layer.get(1).calculateSigmoidOutput(hidden_layer.get(1).calculateNetSum());
        //TODO: UNCOMMENT FOR LOOP AND DELETE THE ABOVE NEURONS
        /*
        //Add hidden neurons (as many as specified in NUMBER_OF_HIDDEN_LAYER_NEURONS) and initialise with random weights
        for(int i=0; i < NUMBER_OF_HIDDEN_LAYER_NEURONS; i++){
            HiddenNeuron hn = new HiddenNeuron(inputs_for_the_network, biasNeuronValue);
            hn.setWeightsOfInputs(weightsOfInputs);
            hidden_layer.add(hn);
        } */
        setHiddenNeuronsList(hidden_layer);
        //------------------------------------------------------------------
        System.out.println(" END OF HIDDEN LAYER CREATION ");

        //Create output layer
        ArrayList<OutputNeuron> output_layer = new ArrayList<>();
        //Add output neurons (as many as specified in NUMBER_OF_OUTPUT_LAYER_NEURONS)
        //Outputs of the hidden layer are inputs to the output layer so we must activate their calculations at least once
        ArrayList<Double> inputsForOutputNeurons = new ArrayList<>();
        for(HiddenNeuron hiddenNeuron: hidden_layer){
            inputsForOutputNeurons.add(hiddenNeuron.getSigmoidOutputOfTheHiddenNeuron());
        }
        System.out.println(" END OF OUTPUT LAYER CREATION ");

        //TODO: The full list of weights MUST BE PROVIDED!!!!!!!!!!!!!!!!!!!
        //TODO: Refactor work: remove the responsibility of knowing the weights from the neurons, now that you have a
        //TODO: (continued ) working for look capable of multiplying the derivatives correctly.

        //TEMPORARY LIST OF WEIGHTS:
        ArrayList<Double> allWeightsBetweenHiddenAndOutputLayer = new ArrayList<>(Arrays.asList(0.4,0.45,0.50,0.55));

        //add two output neurons to the output layer----------------------------------------OUTPUT NEURON LAYER
        OutputNeuron out1 = new OutputNeuron(inputsForOutputNeurons, 0.60, 0.01);
        //give neuron the list of weights
        out1.setWeightsOfOutputsFromHiddenLayer(allWeightsBetweenHiddenAndOutputLayer);
        //add to the output layer and fire
        output_layer.add(out1);
        double ns1 = out1.calculateNetSum();
        double finalOut = out1.calculateSigmoidOutput(ns1);


        OutputNeuron out2 = new OutputNeuron(inputsForOutputNeurons, 0.60, 0.99);
        //give neuron the list of weights
        out2.setWeightsOfOutputsFromHiddenLayer(allWeightsBetweenHiddenAndOutputLayer);
        //Add to the utput layyer and fire
        output_layer.add(out2);
        double ns2 = out2.calculateNetSum();
        double finalOut2 = out2.calculateSigmoidOutput(ns2);
       /*
        ArrayList<Double> weightsOfHiddenLayerOutputs = new ArrayList<>();
        for(HiddenNeuron hiddenNeuron: hidden_layer){
            weightsOfInputs.add(random.nextDouble());
        }
        //add the output neurons to the output layer
        for(int i=0; i < NUMBER_OF_OUTPUT_LAYER_NEURONS; i++){
            OutputNeuron on = new OutputNeuron(inputs_for_the_network, biasNeuronValue);
            on.setHiddenLayerOutputValuesList(inputsForOutputNeurons);
            on.setWeightsOfOutputsFromHiddenLayer(weightsOfHiddenLayerOutputs);
            output_layer.add(on);
        } */

        setOutputNeuronsList(output_layer);

        //Finally, set the target-outputs and actual outputs list
        //It contains a list of target outputs, each coressponding to an output neuron
        for(OutputNeuron o: getOutputNeuronsList()){
            targetOutputsList.add(o.getTargetOutput());
        }

       //Set list of actual outputs. These will be used by the forward-propagation
        actualOutputsList.add(finalOut);
        actualOutputsList.add(finalOut2);
    }

    public ArrayList<Double> fowardPropagate(){
        ArrayList<Double> inputs_for_the_network = getInputsList();
        //TODO: IMPLEMENT FORWARD PROPAGATION!!!!!
        ArrayList<Double> outputs = new ArrayList<>();
        System.out.println("##############################-START-############################################");
        //Hidden layer calculations
        for(HiddenNeuron hiddenNeuron: getHiddenNeuronsList()){
            hiddenNeuron.fire();
            System.out.println("HIDDEN NEURON >>>>>>>>>>>>>>>>>>>> Hidden Neuron fired!!\n");
        }
        //Output layer calculations
        for(OutputNeuron outputNeuron: getOutputNeuronsList()){
            //Fire the output neurons and store the outputs of each within the neuron
            double output = outputNeuron.fire();
            outputs.add(output);
            //Store the output in the neuron for later use in the backpropagation algorithm
            outputNeuron.setActualOutput(output);
            System.out.println("OUTPUT NEURON >>>>>>>>>>>>>>>>>>>> Output Neuron fired!!\n");
        }

        //PRINT TO THE TERMINAL
        for(double d: outputs){
            System.out.println("FORWARD PROPAGATION OUTPUTS: " + d);
        }
        System.out.println("#################################-END-###########################################");

        //Set the outputs list to the new actual-outputs of the output neurons
        actualOutputsList = outputs;

        return outputs;
    }

    //TODO: UNDER CONSTRUCTION===============================================================================================####
    //An implementation of the Backpropagation algorithm based on the calculation of partial derivatives
    public void backpropagate(){
        //TODO: Implement backpropagation algorithm for the network
        //Storage for the updated weights after backpropagation
        ArrayList<Double> newWeightsOfInputs = new ArrayList();
        ArrayList<Double> newWeightsOfOutputsFromHiddenLayer = new ArrayList();

        //Calculate the total output error
        //Note: function also sets the individual output error in the neuron which can be accessed using the [ getOutputError ] getter method
        double totalNetworkOutputError = calculateOutputErrorForEachOutputNeuron(outputNeuronsList);

        //Firstly:
        //Take the partial derivative of the total output error with respect to the output
        // Derivative of the TotalError with respect to the Output multiplied by derivative of the Output with respect to the net sum input of the output neuron
        // Once we have these in a table, we only need to multiply each by the partial derivative of the net input with respect to the appropriate weight connected to the neuron
        //
        //The partial derivative of the total error with respect to a weight:
        //dTotalError/dWeighti = dTotalError/dOuti * dOuti/dNeti * dNeti/dWeighti
        //
        //In the for loop below we calculate the first two because they do not change throughout the backpropagation (dTotalError/dOuti * dOuti/dNeti)
        ArrayList<Double> totalErrorChange_WRT_Output_List_X_net_List = new ArrayList();
        for(OutputNeuron outputNeuron: getOutputNeuronsList()){
            //vars
            double targetOut = outputNeuron.getTargetOutput();
            double actualOut = outputNeuron.getActualOutput();
            //Partial derivative of the total error with respect to the output
            double partialDerivative_O = -(targetOut - actualOut);
            //Partial derivative of the output with respect to the net input
            double partialDerivative_N = (actualOut * (1 - actualOut));
            //We store this because its values will be used in calculations of all the weights and the values will be constant for this part
            double ans = (partialDerivative_O * partialDerivative_N);
            totalErrorChange_WRT_Output_List_X_net_List.add(ans);

            System.out.println(">>> THE CHANGE WITH RESPECT TO THE OUTPUT for this output is: " + partialDerivative_O);
            System.out.println(">>> THE CHANGE WITH RESPECT TO THE OUTPUT for this net input is: " + partialDerivative_N);
        }

        //now for each appropriate weight we can now calculate dNeti/dWeighti and complete the equation for each weight


        ///-----------Pseudocode--------------------
        //Each output neuron has a list of all the weights connecting the output layer to the hidden layer
        ArrayList<Double> weightsBetweenOutputAndHiddenLayers_List = outputNeuronsList.get(0).getWeightsOfOutputsFromHiddenLayer();
        ArrayList<Double> hiddenLayerOutputs_List = outputNeuronsList.get(0).getHiddenLayerOutputValuesList();

        //For each weight, I multiply by the output of the hidden neurons, incrementing the index of the hidden neuron whos output im multiplying by
        //The increment must go up to the number of neurons in the hidden layer and must restart from neuron 1 once the index reaches the total number of hidden neurons.
        int indexOfHiddenNeuronToMultiplyBy = 0; //this is the first hidden neuron
        int totalNumberOfNiddenNeurons = hiddenNeuronsList.size();

        for(double weight: weightsBetweenOutputAndHiddenLayers_List){
            //Reset if greater than total number of hidden neurons
            if(indexOfHiddenNeuronToMultiplyBy >= totalNumberOfNiddenNeurons){
                indexOfHiddenNeuronToMultiplyBy = 0;
            }

            //Completing the equation for each weight ---> dTotalError/dWeighti = dTotalError/dOuti * dOuti/dNeti * dNeti/dWeighti
            double fullWeightError = totalErrorChange_WRT_Output_List_X_net_List.get(indexOfHiddenNeuronToMultiplyBy) * hiddenLayerOutputs_List.get(indexOfHiddenNeuronToMultiplyBy);
            double updatedWeight = weight - (learningRate * fullWeightError);

            System.out.println("THE UPDATED WEIGHT FOR WEIGHT: " + weight + " ====> " + updatedWeight);

            //increment up to the total number of hidden neurons each time and reset when limit reached
            indexOfHiddenNeuronToMultiplyBy = indexOfHiddenNeuronToMultiplyBy + 1;
        }



        ///-----------------------------------------

        //For each weight, calculate the updated weight

        //more to do ...

    }
    //TODO: END OF - UNDER CONSTRUCTION===============================================================================================####

    //This method calculates the individual output error of each neuron and then sums all of them to get the total output error
    public double calculateOutputErrorForEachOutputNeuron(ArrayList<OutputNeuron> output_neurons_list){
        //clear the error list and update with new values
        errorList.clear();

        double totalOutputError = 0;

        for(OutputNeuron outputNeuron: output_neurons_list){
            //Reminder: This outputs correctly. Do not modify.
            double target = getTargetOutputsList().get(output_neurons_list.indexOf(outputNeuron));
            double output = outputNeuron.getActualOutput();
            System.out.println(">>> T - O = " + (target-output));
            //Squared error equation is = 1/2(target - output^2)
            //We use 1.0/2.0 to represent 1/2 because we are dealing with doubles
            double error = ((1.0/2.0) * Math.pow((target - output),2));
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
