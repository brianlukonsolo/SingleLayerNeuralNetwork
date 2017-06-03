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

        //TODO: The full list of weights MUST BE PROVIDED!!!!!!!!!!!!!!!!!!!
        //TODO: Refactor work: remove the responsibility of knowing the weights from the neurons, now that you have a
        //TODO: (continued ) working for look capable of multiplying the derivatives correctly.

        //TEMPORARY LISTS OF WEIGHTS:
        ArrayList<Double> allWeightsBetweenInputAndHiddenLayer = new ArrayList<>(Arrays.asList(0.15, 0.20, 0.25, 0.30));
        ArrayList<Double> allWeightsBetweenHiddenAndOutputLayer = new ArrayList<>(Arrays.asList(0.4, 0.45, 0.50, 0.55));

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


        //Create the hidden layer
        ArrayList<HiddenNeuron> hidden_layer = new ArrayList<>();
        //hidden neuron 1
        hidden_layer.add(new HiddenNeuron(inputs_for_the_network, 0.35));
        hidden_layer.get(0).setWeightsOfInputs(allWeightsBetweenInputAndHiddenLayer);
        //Fire the neuron once or network will fail
        hidden_layer.get(0).calculateSigmoidOutput(hidden_layer.get(0).calculateNetSum());

        //hidden neuron 1
        hidden_layer.add(new HiddenNeuron(inputs_for_the_network, 0.35));
        hidden_layer.get(1).setWeightsOfInputs(allWeightsBetweenInputAndHiddenLayer);
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
        for (HiddenNeuron hiddenNeuron : hidden_layer) {
            inputsForOutputNeurons.add(hiddenNeuron.getSigmoidOutputOfTheHiddenNeuron());
        }
        System.out.println(" END OF OUTPUT LAYER CREATION ");


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
        for (OutputNeuron o : getOutputNeuronsList()) {
            targetOutputsList.add(o.getTargetOutput());
        }

        //Set list of actual outputs. These will be used by the forward-propagation
        actualOutputsList.add(finalOut);
        actualOutputsList.add(finalOut2);
    }

    public ArrayList<Double> fowardPropagate() {
        ArrayList<Double> inputs_for_the_network = getInputsList();
        //TODO: IMPLEMENT FORWARD PROPAGATION!!!!!
        ArrayList<Double> outputs = new ArrayList<>();
        System.out.println("##############################-START-############################################");
        //Hidden layer calculations
        for (HiddenNeuron hiddenNeuron : getHiddenNeuronsList()) {
            hiddenNeuron.fire();
            System.out.println("HIDDEN NEURON >>>>>>>>>>>>>>>>>>>> Hidden Neuron fired!!\n");
        }
        //Output layer calculations
        for (OutputNeuron outputNeuron : getOutputNeuronsList()) {
            //Fire the output neurons and store the outputs of each within the neuron
            double output = outputNeuron.fire();
            outputs.add(output);
            //Store the output in the neuron for later use in the backpropagation algorithm
            outputNeuron.setActualOutput(output);
            System.out.println("OUTPUT NEURON >>>>>>>>>>>>>>>>>>>> Output Neuron fired!!\n");
        }

        //PRINT TO THE TERMINAL
        for (double d : outputs) {
            System.out.println("FORWARD PROPAGATION OUTPUTS: " + d);
        }
        System.out.println("#################################-END-###########################################");

        //Set the outputs list to the new actual-outputs of the output neurons
        actualOutputsList = outputs;

        return outputs;
    }

    //TODO: UNDER CONSTRUCTION===============================================================================================####
    //An implementation of the Backpropagation algorithm based on the calculation of partial derivatives
    public void backpropagate() {
        //TODO: Implement backpropagation algorithm for the network
        //Storage for the updated weights after backpropagation
        ArrayList<Double> updatedWeights_InputsToHiddenLayer = new ArrayList();
        ArrayList<Double> updatedWeights_HiddenLayerToOutputs = new ArrayList();

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
        for (OutputNeuron outputNeuron : getOutputNeuronsList()) {
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

        //Each output neuron has a list of all the weights connecting the output layer to the hidden layer
        ArrayList<Double> weightsBetweenOutputAndHiddenLayers_List = outputNeuronsList.get(0).getWeightsOfOutputsFromHiddenLayer();
        ArrayList<Double> hiddenLayerOutputs_List = outputNeuronsList.get(0).getHiddenLayerOutputValuesList();
        //For each weight, I multiply by the output of the hidden neurons, incrementing the index of the hidden neuron whos output im multiplying by
        //The increment must go up to the number of neurons in the hidden layer and must restart from neuron 1 once the index reaches the total number of hidden neurons.
        int indexOfHiddenNeuronToMultiplyBy = 0; //this is the first hidden neuron
        int totalNumberOfNiddenNeurons = hiddenNeuronsList.size();

        //---------------------------------------------------------------------------------------------------------#
        //This will calculate the updated weights for the connections between the hidden layer and the output layer
        //---------------------------------------------------------------------------------------------------------#
        for (double weight : weightsBetweenOutputAndHiddenLayers_List) {
            //Reset if greater than total number of hidden neurons
            if (indexOfHiddenNeuronToMultiplyBy >= totalNumberOfNiddenNeurons) {
                indexOfHiddenNeuronToMultiplyBy = 0;
            }

            //Completing the equation for each weight ---> dTotalError/dWeighti = dTotalError/dOuti * dOuti/dNeti * dNeti/dWeighti
            double fullWeightError = totalErrorChange_WRT_Output_List_X_net_List.get(indexOfHiddenNeuronToMultiplyBy) * hiddenLayerOutputs_List.get(indexOfHiddenNeuronToMultiplyBy);
            double updatedWeight = weight - (learningRate * fullWeightError);

            //Store the updated weights
            updatedWeights_HiddenLayerToOutputs.add(updatedWeight);

            System.out.println("THE UPDATED WEIGHT FOR WEIGHT: " + weight + " ====> " + updatedWeight);

            //increment up to the total number of hidden neurons each time and reset when limit reached
            indexOfHiddenNeuronToMultiplyBy = indexOfHiddenNeuronToMultiplyBy + 1;
        }


        //Each output neuron has a list of all the weights connecting the output layer to the hidden layer
        ArrayList<Double> weightsBetweenInputAndHiddenLayers_List = hiddenNeuronsList.get(0).getWeightsOfInputs();
        int indexOfInputToMultiplyBy = 0;
        int totalNumberOfInputs = inputsList.size();
        int indexOfOutputLayerWeightToMultiplyBy = 0;
        //This makes sure that we multiply by the correct hidden-output weights even if we increase the number of hidden layer neurons
        int numberOfHiddenNeurons = hiddenNeuronsList.size();
        int indexOfWeightToMultiplyByForDerivative = 0;

        //---------------------------------------------------------------------------------------------------------#
        //This will calculate the updated weights for the connections between the input layer and the hidden layer
        //---------------------------------------------------------------------------------------------------------#
        System.out.println("\n ...Weights between input and hidden layer>>>>" + weightsBetweenInputAndHiddenLayers_List + "\n");
        System.out.println("\n ...Weights between hidden and output layer>>>>" + weightsBetweenOutputAndHiddenLayers_List + "\n");
        double outputError;
        double totalOutputError;

        //For each weight between the input and hidden layer
        int numberOfInputs = getInputsList().size();
        int numberOfHiddenLayerNeurons = getHiddenNeuronsList().size();
        int numberOfOutputLayerNeurons = getOutputNeuronsList().size();
        int inputWeightIndex = 0;
        System.out.println("####[[[ Number of inputs: " + numberOfInputs + " ]]]####");
        int indexShift = 1;

        //int count = 0;
        //int increment = 0;

        //For each weight between the input and hidden layer, I want to find the hidden neurons associated with that weight
        //InputLayer
        /*for (double weight : weightsBetweenInputAndHiddenLayers_List) {
            inputWeightIndex = inputWeightIndex + 1;
            //HiddenLayer
            for(int i = 0; i<numberOfHiddenLayerNeurons ; i++){//HiddenNeuron hiddenNeuron: getHiddenNeuronsList()){
                System.out.println("\nHIDDEN NEURON -=====================================================#### " + getHiddenNeuronsList().indexOf(i));
                System.out.println("Count is : " + count);
                //If the current hidden neuron index == the number of input neurons

                if(increment >= weightsBetweenInputAndHiddenLayers_List.size()){
                    increment = 0;
                }
                System.out.println("WEIGHT: " + weightsBetweenInputAndHiddenLayers_List.get(count));

                count = count + 1;
                if(count >= (weightsBetweenInputAndHiddenLayers_List.size())){
                    System.out.println("----COUNT LIMIT>>>>>>>>>>>>>--" + count);
                    count = 0;
                    break;
                }
            }

        }*/


        int c = 0;
        int shift = 0;

        //for (HiddenNeuron hiddenNeuron : getHiddenNeuronsList()) {
        for (int i = 0; i < numberOfInputs; i++) {
            System.out.println("NEXT INPUT [ [ [----start index =--] ] ]");
            //For each weight
            for (int w = 0; w < weightsBetweenInputAndHiddenLayers_List.size(); w++) {
                if (shift > numberOfInputs) {
                    shift = 1;
                }

                if (c >= numberOfInputs) {
                    c = 0;
                }
                //Loop to print to terminal
                int counter = 0;
                for (int l = 0; l < numberOfInputs; l++) {
                    System.out.println("Weight needed >> " + weightsBetweenInputAndHiddenLayers_List.get(c + shift + counter));
                    counter += numberOfInputs;
                }

                //increment according to number of inputs
                //This stops unnecessary looping because of the difference between the number of weights and the number of inputs
                w = (w + numberOfInputs) * numberOfInputs;

                System.out.println("        ENDPART >> C is = " + c);
            }

            shift = shift + 1;
        }


        //}


        //Finally update ALL the weights in the network
        //TODO: Finally, update all the weights in the network!! (dont forget to give the neurons the updated weights if still giving them this responsibility of knowing the full list)

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
