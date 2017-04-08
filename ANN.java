import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
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
class ANN {

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
    // public final double BETA;
    // public final double LAMDA;

    /**
     * Training iterations. You should use it in your implementation.
     * You can set the value via command line parameter.
     */
    public final int MAX_FAIL = 10;
    public double correct_counter;
    public String [] layerTypes;
    public ArrayList<Mat3D> activationLayers; // Activations
    public ArrayList<Mat3D> zLayers; // Weighted sums 1 to 1 mapping to activationLayers
    public ArrayList<Mat3D> deltaLayers;
    public ArrayList<ArrayList<Mat3D>> filterLayers;
    public ArrayList<ArrayList<Mat3D>> dC_dWeights;
    public ArrayList<ArrayList<Double>> biasWeights;
    public ArrayList<ArrayList<Double>> dC_dBias;
    public ArrayList<ArrayList<Mat3D>> currBestWeights;
    public ArrayList<ArrayList<Double>> currBestBiasWeights;
    public int numLayers;
    public int numLabels;
    public int [] inputDims;

    public int numFail;
    public int TUNE_PERIOD = 10;
    public int MAX_EPOCH = 1000;
    public List<Double> tuneAccuracies;
    public List<Double> tuneErrors;
    public List<Integer> tuneEpochs;
    public int currBestEpoch;
    public double currBestAccuracy;
    public double testAccuracy;
    private static List<String> Category = new ArrayList<>();



    /**
     * Constructor. You should initialize the Perceptron weights in this
     * method. Also, if necessary, you could do some operations on
     * your own variables or objects.
     *
     * @param alpha
     *      The value for initializing learning rate.
     *
     * @param epoch
     *      The value for initializing training iterations.
     *
     * @param num_features
     *      This is the length of input feature vector. You might
     *      use this value to create the input units.
     *
     * @param num_labels
     *      This is the size of label set. You might use this
     *      value to create the output units.
     */
    public ANN(String [] layerTypes, int [][] layerDims, double alpha, int numEpochs, int [] inputDims, int numLabels, int tunePeriod) {

        String[] a = new String[]{"airplanes", "butterfly", "flower", "grand_piano", "starfish", "watch"};
        Category = Arrays.asList(a);

        this.ALPHA = alpha;
        this.MAX_EPOCH = numEpochs;
        this.TUNE_PERIOD = tunePeriod;
        this.inputDims = inputDims;
        this.numLabels = numLabels;

        this.layerTypes = layerTypes;
        this.numLayers = layerTypes.length;
        this.activationLayers = new ArrayList<Mat3D>(this.numLayers);
        this.zLayers = new ArrayList<Mat3D>(this.numLayers);
        this.deltaLayers = new ArrayList<Mat3D>(this.numLayers);
        this.filterLayers = new ArrayList<ArrayList<Mat3D>>(this.numLayers);
        this.dC_dWeights = new ArrayList<ArrayList<Mat3D>>(this.numLayers);
        this.currBestWeights = new ArrayList<ArrayList<Mat3D>>(this.numLayers);
        this.biasWeights = new ArrayList<ArrayList<Double>>(this.numLayers);
        this.currBestBiasWeights = new ArrayList<ArrayList<Double>>(this.numLayers);
        this.dC_dBias = new ArrayList<ArrayList<Double>>(this.numLayers);

        System.out.println(layerDims[0][0]+","+layerDims[0][1]+","+layerDims[0][2]);

        int inputRows = this.inputDims[0];
        int inputCols = this.inputDims[1];
        int inputDepth = this.inputDims[2];
        Mat3D dummyInputLayer = new Mat3D(inputRows,inputCols,inputDepth);

        for (int i = 0;i < this.numLayers;i++) {
            int numRows = layerDims[i][0];
            int numCols = layerDims[i][1];
            int depth = layerDims[i][2];
            if(layerTypes[i].equals("conv")) {
                activationLayers.add(new Mat3D(numRows,numCols,depth));
                zLayers.add(new Mat3D(numRows,numCols,depth));
                deltaLayers.add(new Mat3D(numRows,numCols,depth));
                if(i == 0) {
                    filterLayers.add(generateConvFilterLayer(dummyInputLayer,numRows,numCols,depth));
                    biasWeights.add(generateConvBiasLayerWeights(dummyInputLayer,numRows,numCols,depth));
                }
                else {
                    Mat3D prevActivationLayer = activationLayers.get(i-1);
                    filterLayers.add(generateConvFilterLayer(prevActivationLayer,numRows,numCols,depth));
                    biasWeights.add(generateConvBiasLayerWeights(prevActivationLayer,numRows,numCols,depth));
                }
            }
            else if(layerTypes[i].equals("fullyConnected")) {
                int numHiddenUnits = depth;
                activationLayers.add(new Mat3D(1,1,numHiddenUnits));
                zLayers.add(new Mat3D(1,1,numHiddenUnits));
                deltaLayers.add(new Mat3D(1,1,numHiddenUnits));
                if(i == 0) {
                    filterLayers.add(generateHiddenUnitWeightLayer(dummyInputLayer,numHiddenUnits));
                    biasWeights.add(generateHiddenUnitBiasWeight(dummyInputLayer,numHiddenUnits));
                }
                else {
                    Mat3D prevActivationLayer = activationLayers.get(i-1);
                    filterLayers.add(generateHiddenUnitWeightLayer(prevActivationLayer,numHiddenUnits));
                    biasWeights.add(generateHiddenUnitBiasWeight(prevActivationLayer,numHiddenUnits));

                }
            }
            else if(layerTypes[i].equals("outputLayer")) {
                int numOutputs = depth;
                activationLayers.add(new Mat3D(1,1,numOutputs));
                zLayers.add(new Mat3D(1,1,numOutputs));
                deltaLayers.add(new Mat3D(1,1,numOutputs));
                if(i == 0) {
                    filterLayers.add(generateHiddenUnitWeightLayer(dummyInputLayer,numOutputs));
                    biasWeights.add(generateHiddenUnitBiasWeight(dummyInputLayer,numOutputs));
                }
                else {
                    Mat3D prevActivationLayer = activationLayers.get(i-1);
                    filterLayers.add(generateHiddenUnitWeightLayer(prevActivationLayer,numOutputs));
                    biasWeights.add(generateHiddenUnitBiasWeight(prevActivationLayer,numOutputs));
                }
            }
            else if (layerTypes[i].equalsIgnoreCase("maxpool")) {
                activationLayers.add(new Mat3D(numRows,numCols,depth));
                zLayers.add(new Mat3D(numRows,numCols,depth));
                deltaLayers.add(new Mat3D(numRows,numCols,depth));
                // Add empty array lists for filter and bias weights to keep 1 to 1 correspondance in list
                filterLayers.add(new ArrayList<Mat3D>());
                dC_dWeights.add(new ArrayList<Mat3D>());
                biasWeights.add(new ArrayList<Double>());
                dC_dBias.add(new ArrayList<Double>());
            }
            else {
                System.err.println("initialization error: layer type not recognized");
                System.exit(1);
            }
        }

        // this.output1 = new double[num_hu];
        // this.output = new double[num_labels];
        this.currBestWeights = this.filterLayers;
        this.tuneAccuracies = new ArrayList<Double>();
        this.tuneErrors = new ArrayList<Double>();
        this.tuneEpochs = new ArrayList<Integer>();


    }


    /**
     * Train your Perceptron in this method.
     *
     * @param training_data
     */
    public void train(Dataset dataset, Dataset tune) {

        for(int currEpoch = 0;currEpoch < MAX_EPOCH;currEpoch++) {
            for(Instance inst:dataset.getImages()) {
                backprop(inst);
                updateWeights();
            }
            // Early stopping check
            if(currEpoch % this.TUNE_PERIOD == 0) {
                System.out.println("Epoch: " + currEpoch);
                double currAccuracy = this.test(tune,false);
                this.tuneAccuracies.add(currAccuracy);
                this.tuneErrors.add(1-currAccuracy);
                this.tuneEpochs.add(currEpoch+1);

                if(currAccuracy > this.currBestAccuracy) {
                    this.numFail = 0;
                    this.currBestAccuracy = currAccuracy;
                    this.currBestWeights = this.filterLayers;
                    this.currBestBiasWeights = this.biasWeights;
                    // this.currBestWeights2 = this.weights2;
                    this.currBestEpoch = currEpoch;
                }
                else {
                    this.numFail++;
                    if(this.numFail > this.MAX_FAIL) {
                        System.out.println("Early Stopping. New accuracy: " + currAccuracy + ", is lower than previous model accuracy: " + this.currBestAccuracy);
                        // System.out.println("The best tune accuracy was obtained during epoch: " + this.curr_best_epoch);
                        System.out.println("Early Stopping happens  when the accuracy on tune set decreases for " + this.MAX_FAIL*this.TUNE_PERIOD + " epochs in a row");
                        this.filterLayers = this.currBestWeights;
                        this.biasWeights = this.currBestBiasWeights;
                        break;
                    }
                }
            }
        }

    }
    public double test(Dataset dataset, boolean isTest) {
        double error = 0.0;
        double accuracy = 0.0;
        int correctCount = 0;
        int[][] confMatrix = new int[6][6];

        for (Instance inst : dataset.getImages() ) {
            String outLabel = feedforward(inst);
            String actualLabel = inst.getLabel();
            int outLabelIndex = Category.indexOf(outLabel);
            int actualLabelIndex =  Category.indexOf(actualLabel);
            confMatrix[actualLabelIndex][outLabelIndex] += 1;
            // System.out.println("Predicted: " + outLabel + ", Actual: " + inst.getLabel());

            if(outLabel.equals(inst.getLabel())) {
                correctCount++;
            }
        }
        accuracy = ((double)(correctCount)) / dataset.getImages().size() ;
        if(isTest) {
            System.out.println("Accuracy on Test set: " + accuracy);
        }
        else {
            System.out.println("Accuracy on Tune set: " + accuracy);
        }

        for (int i = 0;i<confMatrix.length;i++) {
            for (int j = 0;j<confMatrix[i].length;j++ ) {
                System.out.print(String.valueOf(confMatrix[i][j]) + '\t');
            }
            System.out.println('\n');
        }



        return accuracy;
    }

    public String feedforward(Instance image) {
        String label = "";
        // System.out.println("feedforward start");
        //String actualLabel = image.getLabel();

        for(int i = 0;i < this.numLayers;i++) {
            String currLayerType = layerTypes[i];
            ArrayList<Mat3D> filters = filterLayers.get(i);
            ArrayList<Double> biases = biasWeights.get(i);
            int actLayerIndex = i;
            if(i == 0) {
                Mat3D input = image.imageMat3;
                if(currLayerType.equals("conv")){
                    feedforwardConv(actLayerIndex, input, filters, biases);
                }
                else if(currLayerType.equals("maxpool")){
                    feedforwardMaxpool(actLayerIndex, input, filters, biases);
                }
                else if(currLayerType.equals("fullyConnected")){
                    feedforwardFullyConnected(actLayerIndex, input, filters, biases);
                }
                else if(currLayerType.equals("outputLayer")){
                    feedforwardOutputLayer(actLayerIndex, input, filters, biases);
                }
            }
            else {
                Mat3D input = activationLayers.get(i-1);
                if(currLayerType.equals("conv")){
                    feedforwardConv(actLayerIndex, input, filters, biases);
                }
                else if(currLayerType.equals("maxpool")){
                    feedforwardMaxpool(actLayerIndex, input, filters, biases);
                }
                else if(currLayerType.equals("fullyConnected")){
                    feedforwardFullyConnected(actLayerIndex, input, filters, biases);
                }
                else if(currLayerType.equals("outputLayer")){
                    feedforwardOutputLayer(actLayerIndex, input, filters, biases);
                }
            }
        }
        // // Code goes here
        int labelIndex = activationLayers.get(this.numLayers-1).getMaxIndex();
        label = Category.get(labelIndex);
        // System.out.println("feedforward end");

        return label;
    }


    public void feedforwardConv(int actLayerIndex, Mat3D input, ArrayList<Mat3D> filters, ArrayList<Double> biases){
        int numActivationLayers = activationLayers.get(actLayerIndex).depth;
        for(int j = 0;j < numActivationLayers;j++) {
            Mat3D currFilter = filters.get(j);
            double currBias = biases.get(j);
            Mat2D zl = input.Mat3conv(currFilter);
            zl.addConstant(currBias);
            Mat2D activation = zl.applyReLU();
            zLayers.get(actLayerIndex).mat3.set(j,zl);
            activationLayers.get(actLayerIndex).mat3.set(j,activation);
        }
    }


    public void feedforwardMaxpool(int actLayerIndex, Mat3D input, ArrayList<Mat3D> filters, ArrayList<Double> biases){
        if((input.numRows % activationLayers.get(actLayerIndex).numRows != 0) || (input.numCols % activationLayers.get(actLayerIndex).numCols != 0) ) {
            System.err.println("feedforward maxpool error: dimension mismatch");
            System.exit(1);
        }
        int numRowElems = (int) (input.numRows / activationLayers.get(actLayerIndex).numRows);
        int numColElems = (int) (input.numCols / activationLayers.get(actLayerIndex).numCols);
        zLayers.set(actLayerIndex,input.Maxpool(numRowElems,numColElems));
        activationLayers.set(actLayerIndex,input.Maxpool(numRowElems,numColElems));
    }

    public void feedforwardFullyConnected(int actLayerIndex, Mat3D input, ArrayList<Mat3D> filters, ArrayList<Double> biases){
        int numActivationLayers = activationLayers.get(actLayerIndex).depth;
        for(int j = 0;j < numActivationLayers;j++) {
            Mat3D currWeights = filters.get(j);
            double currBias = biases.get(j);
            Mat2D zl = input.Mat3conv(currWeights);
            zl.addConstant(currBias);
            Mat2D activation = zl.applyReLU();
            zLayers.get(actLayerIndex).mat3.set(j,zl);
            activationLayers.get(actLayerIndex).mat3.set(j,activation);
        }
    }

    public void feedforwardOutputLayer(int actLayerIndex, Mat3D input, ArrayList<Mat3D> filters, ArrayList<Double> biases){
        int numActivationLayers = activationLayers.get(actLayerIndex).depth;
        for(int j = 0;j < numActivationLayers;j++) {
            Mat3D currWeights = filters.get(j);
            double currBias = biases.get(j);
            Mat2D zl = input.Mat3conv(currWeights);
            zl.addConstant(currBias);
            Mat2D activation = zl.applySigmoid();
            zLayers.get(actLayerIndex).mat3.set(j,zl);
            activationLayers.get(actLayerIndex).mat3.set(j,activation);
        }
        // activationLayers.get(actLayerIndex).applySoftmax();
    }

    /**
     *  backpropagate the gradients generated by comparing output with t_output (target output)
     */
    public void backprop(Instance inst) {
        // System.out.println("backproping");
        String outLabel = feedforward(inst);
        Mat3D y = new Mat3D(1,1,this.numLabels);
        y.mat3.get(Category.indexOf(inst.getLabel())).mat[0][0] = 1;
        Mat3D dActivation = activationLayers.get(this.numLayers-1).applyDSigmoid();

        deltaLayers.set(this.numLayers-1, activationLayers.get(this.numLayers-1).subtract(y).Mat3DDotProduct(dActivation) );

        // For each layer
        for(int l = this.numLayers-1;l > 0;l--) {
            // System.out.println("layer: " + this.layerTypes[l]);
            // We need all deltaLplus1
            Mat3D deltaLplus1 = deltaLayers.get(l); //Indexed by j
            Mat3D deltaL = deltaLayers.get(l-1); //Indexed by j
            Mat3D activationL = activationLayers.get(l-1); //Indexed by j
            Mat3D deltaL_left = new Mat3D(deltaL.numRows,deltaL.numCols,deltaL.depth); //Indexed by j
            ArrayList<Mat3D> weightsLplus1 = filterLayers.get(l); // indexed by j, 3D matrix of weights
            // Get 2D plate indexed j
            for(int j = 0;j < deltaLplus1.depth;j++) {

                Mat2D deltaLplus1_j = deltaLplus1.mat3.get(j);

                if(this.layerTypes[l].equals("conv")) {
                    // System.out.println("Calculating deltaL conv to input, j =" + j);
                    Mat3D weightsLplus1_j_rot180 = weightsLplus1.get(j).rotateByNinetyToLeft().rotateByNinetyToLeft();

                    Mat2D deltaLplus1_j_padded = deltaLplus1_j.padWithZeros(weightsLplus1_j_rot180.numRows-1);
                    for(int k = 0;k < deltaL.depth;k++) {
                        deltaL_left.mat3.get(k).addMat(deltaLplus1_j_padded.Matconv(weightsLplus1_j_rot180.mat3.get(k)));
                    }
                }

                else if(this.layerTypes[l].equals("maxpool") && this.layerTypes[l-1].equals("conv")) {
                    // System.out.println("Calculating deltaL maxpool to conv, j =" + j);

                    // activation layer associated to convolution contains the maxIndices
                    Mat2D activationL_k = activationL.mat3.get(j);
                    for(int r = 0;r < activationL_k.maxIndices.length;r++) {
                        for(int c = 0;c < activationL_k.maxIndices[0].length;c++) {
                            int row = activationL_k.maxIndices[r][c][0];
                            int col = activationL_k.maxIndices[r][c][1];
                            deltaL_left.mat3.get(j).mat[row][col] = deltaLplus1_j.mat[r][c];
                        }
                    }

                }
                else if(this.layerTypes[l].equals("fullyConnected") && this.layerTypes[l-1].equals("maxpool")) {
                    Mat3D weightsLplus1_j = weightsLplus1.get(j);
                    // System.out.println("Calculating deltaL fullyConn to maxpool , j =" + j);
                    // deltaLplus1_j.PrintDims();
                    deltaL_left.addMat(weightsLplus1_j.multiplyConstant(deltaLplus1_j.mat[0][0]));
                }

                else if(this.layerTypes[l].equals("outputLayer") && this.layerTypes[l-1].equals("fullyConnected"))
                {
                    Mat3D weightsLplus1_j = weightsLplus1.get(j);
                    // System.out.println("Calculating deltaL for output to fully, j =" + j);
                    for(int k = 0;k < deltaL.depth;k++) {
                        deltaL_left.mat3.get(k).addMat(deltaLplus1_j.Matconv(weightsLplus1_j.mat3.get(k)));
                    }
                }
                // else {
                //  for(int k = 0;k < deltaL.depth;k++) {
                //      deltaL_left.mat3.get(k).addMat(deltaLplus1_j.Matconv(weightsLplus1_j.mat3.get(k))); 
                //  }
                // }

            }

            Mat3D dActivation2 = activationLayers.get(l-1).applyDReLU();
            deltaLayers.set(l-1, deltaL_left.Mat3DDotProduct(dActivation2));
        }

        // System.out.println("end deltaL for " );s

        for(int l = 0;l < this.numLayers;l++) {
            // activations are the inputs for l==0
            // System.out.println("layer: " + this.layerTypes[l]);
            Mat3D aLminus1;
            if(l == 0) {
                aLminus1 = inst.imageMat3;
            }
            else {
                aLminus1 = this.activationLayers.get(l-1);
            }
            Mat3D aLminus1_rotated = aLminus1.rotateByNinetyToLeft().rotateByNinetyToLeft();
            Mat3D deltaL = deltaLayers.get(l);
            int pad = aLminus1.numRows - deltaL.numRows;
            // System.out.println("aLminus1:");
            // aLminus1.PrintDims();
            // System.out.println("deltaL:");
            // deltaL.PrintDims();
            for(int j = 0;j < deltaL.depth;j++) {
                Mat2D deltaL_j = deltaL.mat3.get(j);
                // for(int i = 0; i< aLminus1.depth; i++) {
                //  Mat2D newaL = aLminus1.mat3.get(i).Mat2DDotProduct(deltaL_j);
                //  aLminus1.mat3.set(i, newaL);    
                // }
                // dC_dWeights.get(l).set(j, aLminus1);
                if(this.layerTypes[l].equals("fullyConnected") || this.layerTypes[l].equals("outputLayer")) {
                    dC_dWeights.get(l).set(j, aLminus1.multiplyConstant(deltaL_j.mat[0][0]));
                    dC_dBias.get(l).set(j, deltaL_j.average());
                }
                else if(this.layerTypes[l].equals("conv")) {
                    Mat2D deltaL_j_padded = deltaL_j.padWithZeros(pad);
                    Mat3D dC_dWeights_l_j = new Mat3D(dC_dWeights.get(l).get(j).numRows,dC_dWeights.get(l).get(j).numCols,dC_dWeights.get(l).get(j).depth);
                    double dC_dBias_l_j = 0.0;
                    for(int k = 0;k < dC_dWeights_l_j.depth;k++){
                        dC_dWeights_l_j.mat3.set(k, deltaL_j_padded.Matconv(aLminus1_rotated.mat3.get(k)));
                        dC_dBias_l_j += deltaL_j_padded.average()/dC_dWeights_l_j.depth;
                    }
                    dC_dWeights.get(l).set(j, dC_dWeights_l_j);
                }
            }
        }

    }

    public void updateWeights() {
        for(int l = 0;l < this.numLayers;l++) {
            // ArrayList<Mat3D> currLayerWeights = this.filterLayers.get(l);
            for(int j = 0;j < this.filterLayers.get(l).size();j++) {
                // Mat3D oldWeights = currLayerWeights.get(j);
                // Mat3D newWeights = oldWeights.addMat(dC_dWeights.get(l).get(j).multiplyConstant(this.ALPHA));
                this.filterLayers.get(l).get(j).addMat(dC_dWeights.get(l).get(j).multiplyConstant(-1*this.ALPHA));
                // currLayerWeights.set(j,newWeights);
            }
        }
    }



    private ArrayList<Integer> getIndexArray(int size) {
        ArrayList<Integer> indeces = new ArrayList<Integer>(size);
        for(int i = 0;i < size;i++) {
            indeces.add(i);
        }
        return indeces;
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
     *  - using sigmoid
     */
    public double Sigmoid(double input){
        return 1.0/(1.0+Math.exp(-1.0*input));
    }


    /**
     * Calculate activation function
     *  - using Rectified linear unit
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
     *  - using sigmoid
     *  - f(x) = f(x) * (1-f(x))
     */
    public double DSigmoid(double sigmoid_out) {
        return sigmoid_out*(1-sigmoid_out);
    }

    /**
     * Calculate derivative of sigmoid activation function
     *  - using Rectified linear unit
     */
    public double DReLU(double relu_out) {
        if(relu_out <= 0) {
            return 0;
        }
        else {
            return 1;
        }
    }

    public ArrayList<Mat3D> generateConvFilterLayer(Mat3D prevLayer, int numRows, int numCols, int depth) {
        ArrayList<Mat3D> filterLayer = new ArrayList<Mat3D>(depth);
        ArrayList<Mat3D> dC_dWeightsLayer = new ArrayList<Mat3D>(depth);
        int filterRows = prevLayer.numRows - numRows + 1;
        int filterCols = prevLayer.numCols - numCols + 1;
        int filterDepth = prevLayer.depth;
        int nin = filterRows * filterCols * filterDepth;
        for (int j = 0;j < depth;j++) {
            Mat3D newFilter = new Mat3D(filterRows,filterCols,filterDepth);
            dC_dWeightsLayer.add(new Mat3D(filterRows,filterCols,filterDepth));
            // newFilter.UniformInit(1.0);
            newFilter.WeightInit(nin);
            filterLayer.add(newFilter);
        }
        dC_dWeights.add(dC_dWeightsLayer);
        return filterLayer;
    }

    public ArrayList<Mat3D> generateHiddenUnitWeightLayer(Mat3D prevLayer, int numHiddenUnits) {
        ArrayList<Mat3D> weightLayer = new ArrayList<Mat3D>(numHiddenUnits);
        ArrayList<Mat3D> dC_dWeightsLayer = new ArrayList<Mat3D>(numHiddenUnits);

        int nin = prevLayer.numRows*prevLayer.numCols*prevLayer.depth;
        for(int j = 0;j < numHiddenUnits;j++){
            Mat3D weights = new Mat3D(prevLayer.numRows,prevLayer.numCols,prevLayer.depth);
            dC_dWeightsLayer.add(new Mat3D(prevLayer.numRows,prevLayer.numCols,prevLayer.depth));
            // weights.UniformInit(2.0);
            weights.WeightInit(nin);
            weightLayer.add(weights);
        }
        dC_dWeights.add(dC_dWeightsLayer);

        return weightLayer;
    }

    /**
     * Bias layer for a conv unit depends on the number of conv filters
     */
    public ArrayList<Double> generateConvBiasLayerWeights(Mat3D prevLayer, int numRows, int numCols, int depth) {
        ArrayList<Double> biasLayer = new ArrayList<Double>(depth);
        ArrayList<Double> dC_dBiasLayer = new ArrayList<Double>(depth);
        int filterRows = prevLayer.numRows - numRows + 1;
        int filterCols = prevLayer.numCols - numCols + 1;
        int filterDepth = prevLayer.depth;
        int nin = filterRows * filterCols * filterDepth;
        Random generator = new Random();
        double variance = (double)(1.0/((double)nin));

        for (int j = 0;j < depth;j++) {
            biasLayer.add(generator.nextGaussian() * variance);
            dC_dBiasLayer.add(0.0);
        }
        dC_dBias.add(dC_dBiasLayer);
        return biasLayer;
    }

    /**
     * Bias layer for a hidden unit only needs a single bias
     */
    public ArrayList<Double> generateHiddenUnitBiasWeight(Mat3D prevLayer, int numHiddenUnits) {

        ArrayList<Double> biasLayer = new ArrayList<Double>(numHiddenUnits);
        ArrayList<Double> dC_dBiasLayer = new ArrayList<Double>(numHiddenUnits);
        int nin = prevLayer.numRows*prevLayer.numCols*prevLayer.depth;
        Random generator = new Random();
        double variance = (double)(1.0/((double)nin));
        for (int i = 0;i < numHiddenUnits;i++ ) {
            biasLayer.add(generator.nextGaussian() * variance);
            dC_dBiasLayer.add(0.0);
        }
        dC_dBias.add(dC_dBiasLayer);

        return biasLayer;
    }

    public void printStats() {
        System.out.println("Artificial Neural Network Statistics: ");
        System.out.println("    Number of layers: " + this.numLayers);
        System.out.println("    Activation layer stats: ");
        int totalNumWeights = 0;
        for(int i = 0;i < this.activationLayers.size();i++) {
            System.out.println("        Weight layer #: " + i);
            System.out.println("            Layer type: " + this.layerTypes[i]);
            System.out.println("            Weight layer depth (Num sets of filters): " + filterLayers.get(i).size());
            int numWeights = 0;
            if(!this.layerTypes[i].equals("maxpool")){
                System.out.println("            Filter stats: ");
                System.out.println("                Filter numRows: " + filterLayers.get(i).get(0).numRows);
                System.out.println("                Filter numCols: " + filterLayers.get(i).get(0).numCols);
                System.out.println("                Filter depth: " + filterLayers.get(i).get(0).depth);
                numWeights = filterLayers.get(i).size() * filterLayers.get(i).get(0).numRows * filterLayers.get(i).get(0).numCols * filterLayers.get(i).get(0).depth;
                System.out.println("            Curr layer num weights: " + numWeights);
            }
            totalNumWeights += numWeights;
            System.out.println("        Layer #: " + i);
            System.out.println("            Layer type: " + this.layerTypes[i]);
            System.out.println("            Layer numRows: " + this.activationLayers.get(i).numRows);
            System.out.println("            Layer numCols: " + this.activationLayers.get(i).numCols);
            System.out.println("            Layer depth: " + this.activationLayers.get(i).depth);
        }

        System.out.println("    Total num weights: " + totalNumWeights);

    }
    // public void writeResults(String prefix) {
    //  String path = "results/";
    //  String filename = prefix + "_results_" + this.ALPHA + "_" + this.num_hu + "_" + this.MAX_EPOCH + "_" + this.BETA + ".csv";
    //  System.out.println("Writing results to file: " + path + filename);

    //  try{
    //      PrintWriter writer = new PrintWriter(path+filename, "UTF-8");
    //      for(int i = 0;i < this.tuneEpochs.size();i++) {
    //          writer.print(this.tuneEpochs.get(i) + ", ");
    //      }
    //      writer.println("");
    //      for(int i = 0;i < this.tuneErrors.size();i++) {
    //          writer.print(this.tuneErrors.get(i) + ", ");
    //      }
    //      writer.println("");
    //      for(int i = 0;i < this.tuneAccuracies.size();i++) {
    //          writer.print(this.tuneAccuracies.get(i) + ", ");
    //      }
    //      writer.println("");
    //      writer.println("Final Test Accuracy: " + this.testAccuracy);
    //      writer.println("Best Epoch: " + this.currBestEpoch);
    //      writer.println("Best Tune Accuracy: " + this.currBestAccuracy);
    //      writer.close();

    //  } catch (IOException e) {
    //      System.err.println("Error: Problem writing outputfile!!!");
    //      System.exit(1);
    //  }
    // }

}