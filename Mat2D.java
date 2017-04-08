import java.util.Random;

/**
 * @ Author: Felipe and Nikhil
 *
 * This is the class for each image instance
 */



class Mat2D {

    // Constructor
    // given the bufferedimage and its class label
    // get the 
    double [][] mat;
    int numRows;
    int numCols;

    // ArrayList<ArrayList<Integer>> maxIndices;
    int [][][] maxIndices;

    public Mat2D(int numRows, int numCols) {
        this.numRows = numRows;
        this.numCols = numCols;
        this.mat = new double[numRows][numCols];
    }

    public Mat2D(double [][] inMat) {
        this.mat = inMat;
        this.numRows = inMat.length;
        this.numCols = inMat[0].length;
    }

    public Mat2D(int [][] inMat) {
        this.numRows = inMat.length;
        this.numCols = inMat[0].length;
        mat = new double[numRows][numCols];

        for (int i = 0;i < this.numRows;i++) {
            for (int j = 0;j < this.numCols;j++ ) {
                this.mat[i][j] = (double) inMat[i][j];
            }
        }
    }

    public void setMat(double [][] inMat) {
        this.mat = inMat;
        this.numRows = inMat.length;
        this.numCols = inMat[0].length;
    }

    public void setMat(int [][] inMat) {
        this.numRows = inMat.length;
        this.numCols = inMat[0].length;
        mat = new double[numRows][numCols];

        for (int i = 0;i < this.numRows;i++) {
            for (int j = 0;j < this.numCols;j++ ) {
                this.mat[i][j] = (double) inMat[i][j];
            }
        }
    }

    public Mat2D Matmul(Mat2D inMat) {
        Mat2D outMat = new Mat2D(numRows, inMat.numCols);
        if(this.numCols != inMat.numRows) {
            System.err.println("matmul error: Matrix dimensions don't match");
            System.exit(1);
        }
        double [] row = new double[this.numCols];
        double currRowEntry = 0.0;
        for (int currRow = 0;currRow < this.numRows;currRow++) {
            row = this.mat[currRow];
            for (int i = 0;i < inMat.numRows;i++) {
                currRowEntry = row[i];
                for (int j = 0;j < inMat.numCols;j++ ) {
                    outMat.mat[currRow][j] += currRowEntry*inMat.mat[i][j];
                }
            }
        }
        return outMat;
    }

    public Mat2D Matconv(Mat2D filter) {
        if((this.numCols < filter.numCols) || (this.numRows < filter.numRows) ) {
            System.err.println("matconv error: Filter dimensions are larger than matrix dimensions");
            System.exit(1);
        }
        Mat2D outMat = new Mat2D(this.numRows - filter.numRows + 1, this.numCols - filter.numCols + 1);
        double localConv = 0.0;
        for (int i = 0; i < outMat.numRows;i++) {
            for (int j = 0; j < outMat.numCols;j++) {
                localConv = 0.0;
                for (int k = 0; k < filter.numRows;k++) {
                    for (int l = 0; l < filter.numCols;l++) {
                        localConv += this.mat[i+k][j+l] * filter.mat[k][l];
                    }
                }
                outMat.mat[i][j] = localConv;
            }
        }

        return outMat;
    }

    // public Mat2D Maxpool(int numRowElems, int numColElems) {
    //  this.maxIndices = new ArrayList<ArrayList<Integer>>();
    //  if((this.numRows % numRowElems) != 0 || (this.numCols % numColElems) != 0) {
    //      System.err.println("maxpool error: invalid num row/col elements");
    //      System.exit(1);
    //  }
    //  int newRowDim = (int) (this.numRows / numRowElems);
    //  int newColDim = (int) (this.numCols / numColElems);
    //  double [][] tmpMat = new double[this.numRows][newColDim];
    //  double [][] newMat = new double[newRowDim][newColDim];
    //  double currMax = -Double.MAX_VALUE;
    //  int maxIndexRow = 0;
    //  int maxIndexColumn = 0;
    //  for(int i = 0; i< this.numRows; i+=2) {
    //      for (int j = 0; j< this.numCols; j+=2) {
    //          ArrayList<Integer> tempList = new ArrayList<Integer>();
    //          for(int inr = i; inr< i+numRowElems; inr++) {
    //              for (int inc = j; inc<j+numColElems;inc++ ){
    //                  if(this.mat[inr][inc] > currMax) {
    //                      currMax = this.mat[inr][inc];
    //                      maxIndexRow = inr;
    //                      maxIndexColumn = inc;
    //                  }

    //              }

    //          }
    //          tempList.add(maxIndexRow);
    //          tempList.add(maxIndexColumn);
    //          maxIndices.add(tempList);
    //          newMat[i/numRowElems][j/numColElems] = currMax;

    //      }
    //  }

    //  Mat2D newMat2D = new Mat2D(newRowDim,newColDim);
    //  newMat2D.mat = newMat;

    //  // this.numRows = newRowDim; 
    //  // this.numCols = newColDim; 
    //  // this.mat = newMat; 

    //  return newMat2D;

    // }

    public Mat2D Maxpool(int numRowElems, int numColElems) {
        if((this.numRows % numRowElems) != 0 || (this.numCols % numColElems) != 0) {
            System.err.println("maxpool error: invalid num row/col elements");
            System.exit(1);
        }
        int newRowDim = (int) (this.numRows / numRowElems);
        int newColDim = (int) (this.numCols / numColElems);
        double [][] tmpMat = new double[this.numRows][newColDim];
        double [][] newMat = new double[newRowDim][newColDim];
        int [][][] tmpMaxIndices = new int[this.numRows][newColDim][2];
        this.maxIndices = new int[newRowDim][newColDim][2];

        int currCol = 0;
        double currMax = -Double.MAX_VALUE;
        for (int i = 0;i < this.numRows;i++){
            currCol = 0;
            for (int j = 0;j < this.numCols;j+=numColElems) {
                int [] currMaxIndices = new int[2];
                for(int k = 0;k < numColElems;k++) {
                    if(currMax < this.mat[i][j+k]) {
                        currMax = this.mat[i][j+k];
                        currMaxIndices[0] = i;
                        currMaxIndices[1] = j+k;
                    }
                }
                tmpMat[i][currCol] = currMax;
                tmpMaxIndices[i][currCol] = currMaxIndices;
                currCol++;
                currMax = -Double.MAX_VALUE;
            }
        }
        int currRow = 0;
        for (int j = 0;j < newColDim;j++) {
            currRow = 0;
            for (int i = 0;i < this.numRows;i += numRowElems) {
                int [] currMaxIndices = new int[2];
                currMaxIndices[1] = tmpMaxIndices[i][j][1];
                for(int k = 0;k < numRowElems;k++) {
                    // currMax = Math.max(currMax, tmpMat[i+k][j]);
                    if(currMax < tmpMat[i+k][j]) {
                        currMax = tmpMat[i+k][j];
                        currMaxIndices[0] = i+k;
                    }
                }
                newMat[currRow][j] = currMax;
                this.maxIndices[currRow][j] = currMaxIndices;
                currRow++;
                currMax = -Double.MAX_VALUE;
            }
        }

        // this.numRows = newRowDim; 
        // this.numCols = newColDim; 
        // this.mat = newMat; 
        Mat2D newMat2D = new Mat2D(newRowDim,newColDim);
        newMat2D.mat = newMat;
        return newMat2D;

    }

    public void UniformInit(double val) {
        // System.out.println("Initializing matrix to all values: " + val);
        for (int i = 0;i < this.numRows;i++) {
            for (int j = 0;j < this.numCols;j++ ) {
                this.mat[i][j] = val;
            }
        }
    }

    public void RandomInit() {
        Random generator = new Random();
        // System.out.println("Randomly initializing matrix");
        for (int i = 0;i < this.numRows;i++) {
            for (int j = 0;j < this.numCols;j++ ) {
                this.mat[i][j] = generator.nextDouble();
            }
        }
    }

    public void WeightInit(int nin) {
        Random generator = new Random();
        double variance = (double)(1.0/((double)nin));
        // System.out.println("Randomly initializing matrix");
        for (int i = 0;i < this.numRows;i++) {
            for (int j = 0;j < this.numCols;j++ ) {
                this.mat[i][j] = generator.nextGaussian() * variance;
            }
        }
    }

    public void Normalize(double factor) {
        for (int i = 0;i < this.numRows;i++) {
            for (int j = 0;j < this.numCols;j++ ) {
                this.mat[i][j] = this.mat[i][j] / factor;
            }
        }
    }


    public void Print(){
        System.out.println("Mat2D: ");
        System.out.println("    numRows: " + this.numRows);
        System.out.println("    numCols: " + this.numCols);
        for (int i = 0;i < this.numRows;i++) {
            for (int j = 0;j < this.numCols;j++ ) {
                System.out.print("  "+mat[i][j] + ", ");
            }
            System.out.println("");
        }
    }

    public void PrintMaxIndices(){
        System.out.println("Mat2D MaxIndices: ");
        for (int i = 0;i < this.maxIndices.length;i++) {
            for (int j = 0;j < this.maxIndices[0].length;j++ ) {
                System.out.print("  ("+this.maxIndices[i][j][0] + ", " + this.maxIndices[i][j][1] + "), ");
            }
            System.out.println("");
        }
    }


    public void PrintDims(){
        System.out.println("Mat2D: ");
        System.out.println("    numRows: " + this.numRows);
        System.out.println("    numCols: " + this.numCols);
    }
    public Mat2D rotateByNinetyToLeft() {
        Mat2D newMat = new Mat2D(this.numRows,this.numCols);

        int e = this.mat.length - 1;
        int c = e / 2;
        int b = e % 2;
        for (int r = c; r >= 0; r--) {
            for (int d = c - r; d < c + r + b; d++) {
                double t   = this.mat[c - r][d];
                newMat.mat[c - r][d] = this.mat[d][e - c + r];
                newMat.mat[d][e - c + r] = this.mat[e - c + r][e - d];
                newMat.mat[e - c + r][e - d] = this.mat[e - d][c - r];
                newMat.mat[e - d][c - r] = t;
            }
        }

        newMat.numRows = newMat.mat.length;
        newMat.numCols = newMat.mat[0].length;

        return newMat;
    }


    public Mat2D padWithZeros(int padSize) {
        int newNumCols = this.numCols + 2*padSize;
        int newNumRows = this.numRows + 2*padSize;
        Mat2D paddedMat = new Mat2D(newNumRows,newNumCols);

        // double [][] paddedMat = new double[newNumRows][newNumCols];

        for(int i = padSize; i < newNumRows-padSize; i++) {
            for (int j = padSize; j < newNumCols-padSize; j++) {
                paddedMat.mat[i][j] = this.mat[i-padSize][j-padSize];
            }
        }
        return paddedMat;
        // this.numRows = newNumRows;
        // this.numCols = newNumCols;
        // this.mat = paddedMat; 
    }

    public Mat2D applySigmoid() {
        Mat2D newMat = new Mat2D(this.numRows,this.numCols);
        for(int i = 0; i<this.numRows; i++) {
            for(int j = 0; j<this.numCols; j++) {
                newMat.mat[i][j] = Sigmoid(this.mat[i][j]);
            }
        }
        return newMat;
    }

    public Mat2D applyReLU() {
        Mat2D newMat = new Mat2D(this.numRows,this.numCols);
        for(int i = 0; i<this.numRows; i++) {
            for(int j = 0; j<this.numCols; j++) {
                newMat.mat[i][j] = ReLU(this.mat[i][j]);
            }
        }
        return newMat;
    }

    public Mat2D applyDSigmoid() {
        Mat2D newMat = new Mat2D(this.numRows,this.numCols);
        for(int i = 0; i<this.numRows; i++) {
            for(int j = 0; j<this.numCols; j++) {
                newMat.mat[i][j] = DSigmoid(this.mat[i][j]);
            }
        }
        return newMat;
    }

    public Mat2D applyDReLU() {
        Mat2D newMat = new Mat2D(this.numRows,this.numCols);
        for(int i = 0; i<this.numRows; i++) {
            for(int j = 0; j<this.numCols; j++) {
                newMat.mat[i][j] = DReLU(this.mat[i][j]);
            }
        }
        return newMat;
    }

    public void addConstant(double val) {
        for(int i = 0; i<this.numRows; i++) {
            for(int j = 0; j<this.numCols; j++) {
                this.mat[i][j] += val;
            }
        }
    }

    public Mat2D multiplyConstant(double val) {
        Mat2D newMat = new Mat2D(this.numRows,this.numCols);
        for(int i = 0; i<this.numRows; i++) {
            for(int j = 0; j<this.numCols; j++) {
                newMat.mat[i][j] = this.mat[i][j] * val;
            }
        }
        return newMat;
    }

    public void addMat(Mat2D rightMat) {
        if((this.numRows != rightMat.numRows) || (this.numCols != rightMat.numCols)) {
            System.err.println("addMat error: num row/col mismatch elements");
            System.exit(1);
        }

        for(int i = 0; i<this.numRows; i++) {
            for(int j = 0; j<this.numCols; j++) {
                this.mat[i][j] += rightMat.mat[i][j];
            }
        }
    }

    public Mat2D subtract(Mat2D rightMat) {
        Mat2D newMat = new Mat2D(this.numRows,this.numCols);
        for(int i = 0; i<this.numRows; i++) {
            for(int j = 0; j<this.numCols; j++) {
                newMat.mat[i][j] = this.mat[i][j] - rightMat.mat[i][j];
            }
        }
        return newMat;
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

    public Mat2D Mat2DDotProduct(Mat2D mat2) {
        Mat2D resultMat = new Mat2D(this.numRows, this.numCols);
        if(this.numCols != mat2.numCols || this.numRows != mat2.numRows) {
            System.err.println("Matrix dot product not possible as dimensions don't match");
        }

        for (int i = 0; i < this.numRows; i++) {
            for (int j = 0; j < this.numCols; j++) {
                resultMat.mat[i][j] = this.mat[i][j] * mat2.mat[i][j];
            }
        }

        return resultMat;
    }

    public double average() {
        double sum = 0;
        for (int i = 0; i<this.numRows; i++) {
            for(int j = 0; j< this.numCols; j++) {
                sum += this.mat[i][j];
            }
        }
        sum /= (this.numRows * this.numCols);
        return sum;
    }


}