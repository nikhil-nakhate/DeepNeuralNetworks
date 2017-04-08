import java.util.ArrayList;

/**
 * @ Author: Felipe and Nikhil
 *
 * This is the class for each image instance
 */


class Mat3D {

    // Constructor
    // given the bufferedimage and its class label
    // get the 
    ArrayList<Mat2D> mat3;
    int numRows;
    int numCols;
    int depth;

    // ArrayList<ArrayList<ArrayList<Integer>>> maxIndices;
    // int [][][][] maxIndeces;

    public Mat3D(int numRows, int numCols, int depth) {
        this.numRows = numRows;
        this.numCols = numCols;
        this.depth = depth;
        mat3 = new ArrayList<Mat2D>(depth);
        for (int i = 0;i < depth;i++) {
            mat3.add(new Mat2D(numRows,numCols));
        }

    }

    public Mat2D Mat3conv(Mat3D filter) {

        if((this.depth != filter.depth) ) {
            System.err.println("Mat3Dconv error: Not equal depths");
            System.exit(1);
        }
        Mat2D outMat = new Mat2D(this.numRows - filter.numRows + 1, this.numCols - filter.numCols + 1);
        Mat2D tmpMat = new Mat2D(this.numRows - filter.numRows + 1, this.numCols - filter.numCols + 1);
        double localConv = 0.0;
        for (int sliceIndex = 0;sliceIndex < this.depth;sliceIndex++) {
            tmpMat = this.mat3.get(sliceIndex).Matconv(filter.mat3.get(sliceIndex));
            for (int i = 0; i < outMat.numRows;i++) {
                for (int j = 0; j < outMat.numCols;j++) {
                    outMat.mat[i][j] += tmpMat.mat[i][j];
                }
            }
        }

        return outMat;
    }

    public double Mat3dot(Mat3D leftMat){
        if((this.numRows != leftMat.numRows) || (this.numCols != leftMat.numCols) || (this.depth != leftMat.depth) ) {
            System.err.println("Mat3dot error: Not equal dimensions");
            System.exit(1);
        }
        Mat2D outMat = this.Mat3conv(leftMat);
        return outMat.mat[0][0];
    }

    // public Mat3D Maxpool(int numRowElems, int numColElems) {
    //  this.maxIndices = new ArrayList<>();
    //  if((this.numRows % numRowElems) != 0 || (this.numCols % numColElems) != 0) {
    //      System.err.println("maxpool error: invalid num row/col elements");
    //      System.exit(1);
    //  }
    //  int newRowDim = (int) (this.numRows / numRowElems);
    //  int newColDim = (int) (this.numCols / numColElems);
    //  Mat3D newMat3D = new Mat3D(newRowDim,newColDim,this.depth); 
    //  for(int i = 0;i < this.depth;i++) {
    //      newMat3D.mat3.set(i,mat3.get(i).Maxpool(numRowElems,numColElems));

    //  }
    //  for (int i = 0; i<this.depth; i++) {
    //      maxIndices.add(this.mat3.get(i).maxIndices);
    //  }

    //  return newMat3D;
    // }

    public Mat3D Maxpool(int numRowElems, int numColElems) {
        if((this.numRows % numRowElems) != 0 || (this.numCols % numColElems) != 0) {
            System.err.println("maxpool error: invalid num row/col elements");
            System.exit(1);
        }
        int newRowDim = (int) (this.numRows / numRowElems);
        int newColDim = (int) (this.numCols / numColElems);
        Mat3D newMat3D = new Mat3D(newRowDim,newColDim,this.depth);
        for(int i = 0;i < this.depth;i++) {
            newMat3D.mat3.set(i,mat3.get(i).Maxpool(numRowElems,numColElems));
        }
        return newMat3D;
    }

    public void UniformInit(double val) {
        // System.out.println("Initializing matrix to all values: " + val);
        for(Mat2D mat : this.mat3) {
            mat.UniformInit(val);
        }
    }

    public void RandomInit() {
        // System.out.println("Initializing matrix to random values: ");
        for(Mat2D mat : this.mat3) {
            mat.RandomInit();
        }
    }

    public void WeightInit(int nin) {
        for(Mat2D mat : this.mat3) {
            mat.WeightInit(nin);
        }
    }

    public void Normalize(double factor) {
        for(Mat2D mat : this.mat3) {
            mat.Normalize(factor);
        }
    }

    public void Print(){
        System.out.println("Mat3D: ");
        System.out.println("    numRows: " + this.numRows);
        System.out.println("    numCols: " + this.numCols);
        System.out.println("    depth: " + this.depth);
        for (int i = 0;i < this.depth;i++) {
            mat3.get(i).Print();
        }
    }

    public Mat3D rotateByNinetyToLeft() {
        // System.out.println("Mat3D: ");
        // System.out.println(" numRows: " + this.numRows);
        // System.out.println(" numCols: " + this.numCols);
        // System.out.println(" depth: " + this.depth);
        Mat3D newMat = new Mat3D(this.numRows,this.numCols,this.depth);
        for (int i = 0;i < this.depth;i++) {
            newMat.mat3.set(i,this.mat3.get(i).rotateByNinetyToLeft());
        }
        return newMat;
    }


    public Mat3D padWithZeros(int padSize) {
        int newNumCols = this.numCols + 2*padSize;
        int newNumRows = this.numRows + 2*padSize;
        Mat3D paddedMat = new Mat3D(newNumRows,newNumCols,this.depth);
        // System.out.println("Mat3D: ");
        // System.out.println(" numRows: " + this.numRows);
        // System.out.println(" numCols: " + this.numCols);
        // System.out.println(" depth: " + this.depth);
        for (int i = 0;i < this.depth;i++) {
            paddedMat.mat3.set(i,this.mat3.get(i).padWithZeros(padSize));
            // mat.padWithZeros(padSize);
        }
        return paddedMat;
    }


    public Mat3D applySigmoid() {
        Mat3D newMat = new Mat3D(this.numRows,this.numCols,this.depth);
        for(int i = 0;i < this.depth;i++) {
            newMat.mat3.set(i,this.mat3.get(i).applySigmoid());
        }
        return newMat;
    }

    public Mat3D applyReLU() {
        Mat3D newMat = new Mat3D(this.numRows,this.numCols,this.depth);
        for(int i = 0;i < this.depth;i++) {
            newMat.mat3.set(i,this.mat3.get(i).applyReLU());
        }
        return newMat;
    }

    public Mat3D applyDSigmoid() {
        // Mat3D newMat = new Mat3D(this.numRows,this.numCols,this.depth);
        // for(Mat2D mat : newMat.mat3) {
        //  mat.applyDSigmoid();
        // }    
        Mat3D newMat = new Mat3D(this.numRows,this.numCols,this.depth);
        for(int i = 0;i < this.depth;i++) {
            newMat.mat3.set(i,this.mat3.get(i).applyDSigmoid());
        }

        return newMat;
    }

    public Mat3D applyDReLU() {
        // Mat3D newMat = new Mat3D(this.numRows,this.numCols,this.depth);
        // for(Mat2D mat : newMat.mat3) {
        //  mat.applyDReLU();
        // }    
        Mat3D newMat = new Mat3D(this.numRows,this.numCols,this.depth);
        for(int i = 0;i < this.depth;i++) {
            newMat.mat3.set(i,this.mat3.get(i).applyDReLU());
        }
        return newMat;
    }

    public void addConstant(double val) {
        for(Mat2D mat : this.mat3) {
            mat.addConstant(val);
        }
    }

    public Mat3D multiplyConstant(double val) {
        Mat3D newMat = new Mat3D(this.numRows,this.numCols,this.depth);
        for(int i = 0;i < this.depth;i++) {
            newMat.mat3.set(i, this.mat3.get(i).multiplyConstant(val));
        }
        return newMat;
    }

    public void addMat(Mat3D rightMat) {
        if((this.numRows != rightMat.numRows) || (this.numCols != rightMat.numCols) || (this.depth != rightMat.depth) ) {
            System.err.println("addMat error: Not equal dimensions");
            System.exit(1);
        }
        int layerIndex = 0;
        for(Mat2D mat : this.mat3) {
            mat.addMat(rightMat.mat3.get(layerIndex));
            layerIndex++;
        }

    }

    public Mat3D subtract(Mat3D rightMat) {
        Mat3D newMat = new Mat3D(this.numRows,this.numCols,this.depth);
        for(int i = 0;i < this.depth;i++) {
            newMat.mat3.set(i, this.mat3.get(i).subtract(rightMat.mat3.get(i)));
        }
        return newMat;
    }

    public void applySoftmax() {
        int depth = this.mat3.size();
        double sum = 0;
        int start = 0;
        for (int i = start; i < start + depth; i++) {
            mat3.get(i).mat[0][0] = BoundMath.exp(mat3.get(i).mat[0][0]);
            sum += mat3.get(i).mat[0][0];
        }
        if(Double.isNaN(sum) || sum < 0.0000000000001) {
            for (int i = start; i < start + depth; i++) {
                mat3.get(i).mat[0][0]  = 1.0/depth;
            }
        } else {
            for (int i = start; i < start + depth; i++) {
                mat3.get(i).mat[0][0]  = mat3.get(i).mat[0][0] / sum;
            }
        }
    }

    public int getMaxIndex() {
        if(this.numRows != 1 && this.numCols != 1) {
            System.err.println("This doesnt correspond to an output layer dimension");
        }
        int outputSize = this.depth;
        double maxValue  = -99999.0;
        double[][] confMatrix;
        int maxIndex = 0;
        for(int maxx = 0; maxx<outputSize; maxx++) {
            if(maxValue < this.mat3.get(maxx).mat[0][0]) {
                maxValue = this.mat3.get(maxx).mat[0][0];
                maxIndex = maxx;
            }
        }
        return maxIndex;
    }

    public Mat3D Mat3DDotProduct(Mat3D mat2) {
        Mat3D resultMat = new Mat3D(this.numRows, this.numCols, this.depth);
        if(this.depth != mat2.depth) {
            System.err.println("Matrix dot product not possible as dimensions don't match");
        }
        for (int i = 0; i < this.depth; i++) {
            resultMat.mat3.set(i, this.mat3.get(i).Mat2DDotProduct(mat2.mat3.get(i)));
        }
        return resultMat;
    }

    public void PrintDims(){
        System.out.println("Mat3D: ");
        System.out.println("    numRows: " + this.numRows);
        System.out.println("    numCols: " + this.numCols);
        System.out.println("    depth: " + this.depth);
    }

}