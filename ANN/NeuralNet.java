import java.util.List;
import java.util.ArrayList;
import java.lang.Math;
import java.util.Arrays;
import java.util.Collections;
import java.io.PrintWriter;
import java.io.IOException;
import java.util.Random;

/**
 * You should implement your Perceptron in this class. 
 * Any methods, variables, or secondary classes could be added, but will 
 * only interact with the methods or variables in this framework.
 *
 * You must add code for at least the 3 methods specified below. Because we
 * don't provide the weights of the Perceptron, you should create your own 
 * data structure to store the weights.
 *
 */
public class NeuralNet {

    /**
     * The initial value for ALL weights in the Perceptron.
     * We fix it to 0, and you CANNOT change it.
     */
    public final double INIT_WEIGHT = 0.0;

    /**
     * Learning rate value. You should use it in your implementation.
     * You can set the value via command line parameter.
     */
    public final double ALPHA;
    public final double BETA;
    public final double LAMDA;

    /**
     * Training iterations. You should use it in your implementation.
     * You can set the value via command line parameter.
     */
    public final int MAX_FAIL = 10;
    public double correct_counter;
    public double[][] weights1; // n_hu x 357 + 1(num_features)
    public double[][] weights2; // 3 (num_labels) x n_hu + 1
    public double[][] curr_best_weights1; // Weights with best tune accuracy
    public double[][] curr_best_weights2;// Weights with best tune accuracy
    public double[] output1; // n_hu
    public double[] output2; // 3 (num_labels)
    public double[] output; // 3 (num_labels)
    public double[] deltaj; // n_hu
    public double[] deltak; // n_hu
    public int num_features;
    public int num_labels;
    public int num_hu;
    public int num_fail;
    public int TUNE_PERIOD = 10;
    public int MAX_EPOCH = 1000;
    public List<Double> tune_accuracies;
    public List<Double> tune_errors;
    public List<Integer> tune_epochs;
    public int curr_best_epoch;
    public double curr_best_accuracy;
    public double test_accuracy;
    public double [] prevDeltaWeights2;
    public double [] prevDeltaWeights1;
    //private int seed = 30;
    private Random generator = new Random();
    public char [] possible_labels = {'_','e','h'};


    /**
     * Constructor. You should initialize the Perceptron weights in this
     * method. Also, if necessary, you could do some operations on
     * your own variables or objects.
     *
     * @param alpha
     * 		The value for initializing learning rate.
     *
     * @param epoch
     * 		The value for initializing training iterations.
     *
     * @param num_features
     * 		This is the length of input feature vector. You might
     * 		use this value to create the input units.
     *
     * @param num_labels
     * 		This is the size of label set. You might use this
     * 		value to create the output units.
     */
    public NeuralNet(double alpha, int epoch, int num_features, int num_labels, int num_hu, int tune_period, double beta, double lamda) {

        this.BETA = beta;
        this.LAMDA = lamda;
        this.ALPHA = alpha;
        this.MAX_EPOCH = epoch;
        this.TUNE_PERIOD = tune_period;
        this.num_features = num_features;
        this.num_labels = num_labels;
        this.num_hu = num_hu;

        this.weights1 = new double[num_hu][num_features + 1]; // +1 for bias
        this.weights2 = new double[num_labels][num_hu + 1]; // +1 for bias
        this.curr_best_weights1 = new double[num_hu][num_features + 1]; // +1 for bias
        this.curr_best_weights2 = new double[num_labels][num_hu + 1]; // +1 for bias
        this.output1 = new double[num_hu];
        this.output2 = new double[num_labels];
        this.output = new double[num_labels];
        this.deltaj = new double[num_hu];
        this.deltak = new double[num_labels];
        this.tune_accuracies = new ArrayList<Double>();
        this.tune_errors = new ArrayList<Double>();
        this.tune_epochs = new ArrayList<Integer>();
        this.prevDeltaWeights2 = new double[num_hu + 1];
        this.prevDeltaWeights1 = new double[num_features + 1];

		/*
		 * num_labels columns, one for each digit.
		 * one extra col (first one) for the bias input and bias weight.
		 */

        for(int i = 0;i < num_hu;i++) {
            for(int j = 0;j < num_features+1;j++) {
                double random = generator.nextDouble();
                this.weights1[i][j] = random;

            }
        }
        for(int i = 0;i < num_labels;i++) {
            for(int j = 0;j < num_hu+1;j++) {
                double random = generator.nextDouble();
                this.weights2[i][j] = random;
            }
        }
        // this.curr_best_weights = new double[num_labels][num_features+1];

        // System.out.println("Num labels: " + num_labels);
        // System.out.println("Num features: " + num_features);
        // System.out.println("Num hu: " + num_hu);

    }

    /**
     * Train your Perceptron in this method.
     *
     * @param training_data
     */
    public void train(ProteinDataset dataset) {
        this.curr_best_accuracy = test(dataset.proteins_tune,false);
        this.curr_best_weights1 = this.weights1;
        this.curr_best_weights2 = this.weights2;
        // System.out.println("Accuracy for random weights = " + this.curr_best_accuracy);
        this.tune_accuracies.add(this.curr_best_accuracy);
        this.tune_errors.add(1 - this.curr_best_accuracy);
        this.tune_epochs.add(0);
        this.curr_best_epoch = 0;

        int tune_counter = 0;
        for(int curr_epoch = 0;curr_epoch < MAX_EPOCH;curr_epoch++) {
            // System.out.println("Curr epoch: " + curr_epoch);

            Collections.shuffle(dataset.proteins_train,new Random());

            for(Protein p : dataset.proteins_train) {
                // System.out.println(p.aa.size());
                ArrayList<Integer> window_indeces = getIndexArray(p.aa.size() - 16);
                Collections.shuffle(window_indeces,new Random());
                for (int index : window_indeces) {
                    int start_index = index*21;
                    int end_index = start_index + num_features;
                    double [] features = Arrays.copyOfRange(p.features, start_index, end_index);
                    double [] t_output = p.labels[index]; // target output
                    feedforward(features);
                    backprop(t_output, features);

                }

            }
            if(curr_epoch % this.TUNE_PERIOD == 0) {
                double curr_accuracy = test(dataset.proteins_tune,false);
                this.tune_accuracies.add(curr_accuracy);
                this.tune_errors.add(1-curr_accuracy);
                this.tune_epochs.add(curr_epoch+1);

                if(curr_accuracy > this.curr_best_accuracy) {
                    // System.out.println("Modifyin best weights");
                    this.num_fail = 0;
                    this.curr_best_accuracy = curr_accuracy;
                    this.curr_best_weights1 = this.weights1;
                    this.curr_best_weights2 = this.weights2;
                    this.curr_best_epoch = curr_epoch;
                }
                else {
                    this.num_fail++;
                    if(this.num_fail > this.MAX_FAIL) {
                        System.out.println("Early Stopping. New accuracy: " + curr_accuracy + ", is lower than previous model accuracy: " + this.curr_best_accuracy);
                        // System.out.println("The best tune accuracy was obtained during epoch: " + this.curr_best_epoch);
                        System.out.println("Early Stopping happens  when the accuracy on tune set decreases for " + this.MAX_FAIL*this.TUNE_PERIOD + " epochs in a row");
                        this.weights1 = this.curr_best_weights1;
                        this.weights2 = this.curr_best_weights2;
                        break;
                    }
                }

            }

        }
        this.weights1 = this.curr_best_weights1;
        this.weights2 = this.curr_best_weights2;

    }
    public double test(List<Protein> proteins, boolean isTest) {

        double correct = 0;
        double total_aa = 0;


        for(Protein p : proteins) {
            int counter = 0;
            for(int i = 8;i < p.aa.size() - 8;i++) {
                int start_index = counter*21;
                int end_index = start_index + num_features;
                double [] features = Arrays.copyOfRange(p.features, start_index, end_index);
                double [] t_output = p.labels[counter]; // target output
                // System.out.println(i + " " + features.length);
                counter++;

                int max_index = feedforward(features);
                if(t_output[max_index] == 1) {
                    correct += 1.;
                }
                total_aa += 1.;

                if(isTest) {
                    System.out.println(this.possible_labels[max_index]);
                }
            }
        }
        double accuracy = correct / total_aa;
        if(!isTest){
            System.out.println("tune accuracy = " + accuracy);
        }

        this.test_accuracy = accuracy;

        return accuracy;
    }

    private ArrayList<Integer> getIndexArray(int size) {
        ArrayList<Integer> indeces = new ArrayList<Integer>(size);
        for(int i = 0;i < size;i++) {
            indeces.add(i);
        }
        return indeces;
    }

    private int feedforward(double [] features) {
        // For each hidden unit
        //System.out.println(weights1[0].length);
        for(int i = 0;i < weights1.length;i++) {
            double input_hu = 0.;
            // for each input feature to the hu
            for(int j = 0;j < weights1[0].length - 1;j++) {
                input_hu += weights1[i][j] * features[j];
            }
            // Add bias
            input_hu += weights1[i][weights1[0].length - 1];
            output1[i] = Sigmoid(input_hu);
            //output1[i] = ReLU(input_hu);
        }

        for(int i = 0;i < weights2.length;i++) {
            double input_out_layer = 0.;
            for(int j = 0;j < weights2[0].length - 1;j++) {
                input_out_layer += weights2[i][j] * output1[j];
            }
            // Add bias
            input_out_layer += weights2[i][weights2[0].length - 1];
            output2[i] = Sigmoid(input_out_layer);
        }

        int max_index = getMaxIndex(output2);
        output = new double[num_labels];
        output[max_index] = 1;

        return max_index;

    }

    /**
     *	backpropagate the gradients generated by comparing output with t_output (target output)
     */
    private void backprop(double [] t_output, double [] features) {

        double [] sum_delta_w2 = new double[num_hu+1];
        // calculate the delta for output layer
        for(int i = 0;i < deltak.length;i++) {
            deltak[i] = -1*output2[i]*(1 - output2[i])*(t_output[i] - output2[i]);
            // deltak[i] = output[i]*(1 - output[i])*(t_output[i] - output[i]);
            // deltak[i] = (t_output[i] - output[i]);
        }


        for(int i = 0; i < output.length; i++) {
            for (int j = 0; j < output1.length; j++) {
                // sum_delta_w2[j] += deltak[i] * weights2[i][j];
                sum_delta_w2[j] += deltak[i] * weights2[i][j];
            }
        }

        for (int i = 0;i < deltaj.length;i++) {
             deltaj[i] = output1[i] * (1 - output1[i]) * sum_delta_w2[i];
//            if(output1[i] > 0) {
//                deltaj[i] = sum_delta_w2[i];
//            }
//            else {
//                deltaj[i] = 0;
//            }
        }

        for (int i = 0; i < output.length; i++) {
            //weights2[i] = updateWeights2(deltak[i], weights2[i]);
            double[] new_array = updateWeights2(deltak[i], weights2[i]);
            for(int j = 0; j<new_array.length; j++) {
                weights2[i][j] = new_array[j];
            }
            //System.out.println(weights2[i].length);
            weights2[i][weights2[i].length-1] += ALPHA * deltak[weights2.length-1]; // update bias
            // weights2[i][weights2[i].length-1] += BETA * prevDeltaWeights2[prevDeltaWeights2.length-1]; // momentum
            // weights2[i][weights2[i].length-1] -= LAMDA * ALPHA * weights2[i][weights2[i].length-1]; // weight decay
        }
        // prevDeltaWeights2[prevDeltaWeights2.length-1] = deltak[weights2.length-1]; // save delta for momentum

        for (int i = 0; i < output1.length; i++) {
            double[] new_array = updateWeights1(deltaj[i], weights1[i], features);
            for(int j = 0; j<new_array.length; j++) {
                weights1[i][j] = new_array[j];
            }
            //System.out.println(weights1[i].length);
            weights1[i][weights1[i].length  -1] += ALPHA * deltaj[weights1.length-1]; // update bias
            //weights1[i][weights1[i].length  -1] += BETA * prevDeltaWeights1[prevDeltaWeights1.length-1]; // moemntum
            // weights1[i][weights1[i].length  -1] -= LAMDA * ALPHA * weights1[i][weights1[i].length-1]; // weight decay
        }
        // prevDeltaWeights1[prevDeltaWeights1.length-1] = deltaj[weights1.length-1]; // save delta for momentum

    }
    /**
     * Update HU to Output Weights
     */
    private double [] updateWeights2(double delta, double [] currWeight2) {
        //this.weights2 = new double[num_labels][num_hu + 1];
        double [] updatedWeights2 = new double[num_hu + 1];
        for (int i = 0; i < num_hu; i++) {
            double deltaWeight = ALPHA * delta * output1[i];
            updatedWeights2[i] = currWeight2[i] + deltaWeight;
            // updatedWeights2[i] = updatedWeights2[i]	+ BETA * prevDeltaWeights2[i]; // Momentum
            // updatedWeights2[i] = updatedWeights2[i] - LAMDA * ALPHA * currWeight2[i]; // weight decay
            prevDeltaWeights2[i] = deltaWeight;
        }
        return updatedWeights2;
    }

    private double [] updateWeights1(double delta, double [] currWeight1, double [] features) {
        double [] updatedWeights1 = new double[features.length + 1];
        for (int i = 0; i < num_features; i++) {
            double deltaWeight = ALPHA * delta * features[i];
            updatedWeights1[i] = currWeight1[i] + deltaWeight;
            // updatedWeights1[i] = updatedWeights1[i] + BETA * prevDeltaWeights1[i]; // Momentum
            // updatedWeights1[i] = updatedWeights1[i] - LAMDA * ALPHA * currWeight1[i]; // weight decay
            prevDeltaWeights1[i] = deltaWeight;
        }
        return updatedWeights1;
    }

    public int getMaxIndex(double [] arr) {
        int max_index = 0;
        for(int i = 0;i < arr.length;i++) {
            if(arr[i] > arr[max_index]) {
                max_index = i;
            }
        }
        return max_index;
    }


    /**
     * Calculate activation function
     * 	- using sigmoid
     */
    public double Sigmoid(double input){
        return 1.0/(1.0+Math.exp(-1.0*input));
    }


    /**
     * Calculate activation function
     * 	- using Rectified linear unit
     *  - f(x) = max(0,x)
     */
    public double ReLU(double input){
        if(input <= 0) {
            return 0;
        }
        return input;
    }

    /**
     * Calculate derivative of sigmoid activation function
     * 	- using sigmoid
     *  - f(x) = f(x) * (1-f(x))
     */
    public double DSigmoid(double sigmoid_out) {
        return sigmoid_out*(1-sigmoid_out);
    }

    /**
     * Calculate derivative of sigmoid activation function
     * 	- using Rectified linear unit
     */
    public double DReLU(double relu_out) {
        if(relu_out <= 0) {
            return 0;
        }
        else {
            return 1;
        }
    }

    public void writeResults(String prefix) {
        String path = "results/";
        String filename = prefix + "_results_" + this.ALPHA + "_" + this.num_hu + "_" + this.MAX_EPOCH + "_" + this.BETA + ".csv";
        System.out.println("Writing results to file: " + path + filename);

        try{
            PrintWriter writer = new PrintWriter(path+filename, "UTF-8");
            for(int i = 0;i < this.tune_epochs.size();i++) {
                writer.print(this.tune_epochs.get(i) + ", ");
            }
            writer.println("");
            for(int i = 0;i < this.tune_errors.size();i++) {
                writer.print(this.tune_errors.get(i) + ", ");
            }
            writer.println("");
            for(int i = 0;i < this.tune_accuracies.size();i++) {
                writer.print(this.tune_accuracies.get(i) + ", ");
            }
            writer.println("");
            writer.println("Final Test Accuracy: " + this.test_accuracy);
            writer.println("Best Epoch: " + this.curr_best_epoch);
            writer.println("Best Tune Accuracy: " + this.curr_best_accuracy);
            writer.close();

        } catch (IOException e) {
            System.err.println("Error: Problem writing outputfile!!!");
            System.exit(1);
        }
    }

}