/**
 * Created by nikhil on 1/30/17.
 */

import java.util.*;
import java.io.*;

////////////////////////////////////////////////////////////////////////////
//
// Lab1, Perceptron

//
////////////////////////////////////////////////////////////////////////////


/*
   To run after compiling, type:

java Lab1 <trainsetFilename> <tunesetFilename> <testsetFilename>

*/

public class Lab1 {

    public static void main(String[] args) {
        if (args.length != 3) {
            System.err.println("You must call Lab1 as " +
                    "follows:\n\njava Lab1 " +
                    "<trainsetFilename> <tunesetFilename> <testsetFilename>\n");
            System.exit(1);
        }

        // Read the file names.
        String trainset = args[0];
        String tuneset  = args[1];
        String testset  = args[2];

        // Read in the examples from the files.
        ListOfExamples trainExamples = new ListOfExamples();
        ListOfExamples tuneExamples = new ListOfExamples();
        ListOfExamples testExamples  = new ListOfExamples();
        if (!trainExamples.ReadInExamplesFromFile(trainset) ||
                !tuneExamples.ReadInExamplesFromFile(tuneset) ||
                !testExamples.ReadInExamplesFromFile(testset)) {
            System.err.println("Something went wrong reading the datasets ... " +
                    "giving up.");
            System.exit(1);
        }
        else { /* The following is included so you can see the data organization.
         You'll need to REPLACE it with code that:

          1) uses the TRAINING SET of examples to build a decision tree

          2) prints out the induced decision tree (using simple, indented
             ASCII text)

          3) categorizes the TESTING SET using the induced tree, reporting
             which examples were INCORRECTLY classified, as well as the
             FRACTION that were incorrectly classified.
             Just print out the NAMES of the examples incorrectly classified
             (though during debugging you might wish to print out the full
             example to see if it was processed correctly by your decision
             tree)
      */

            int inputNodes = (trainExamples.getNumberOfFeatures()+1);
            //Arrays for inputs and weights of perceptron
            double inputs[] = new double[inputNodes];
            double weights[] = new double[inputNodes];
            double maxTuneInputs[] = new double[inputNodes];
            double maxTuneWeights[] = new double[inputNodes];

            int i,j;
            //Initializing all weights to 0
            for (i=0;i<trainExamples.getNumberOfFeatures()+1;i++) {
                weights[i] = 0;
            }
            double maxTuneAccuracy=Integer.MIN_VALUE, testAtMaxTuneAccuracy=Integer.MIN_VALUE;
            int maxEpoch=Integer.MIN_VALUE;
            int count=0;
            Random rand = new Random(); //Introduced for randomizing the train examples

            System.out.println("Results are printed after every 50 epochs, upto 1000 epochs");
            for (i=0;i<1000;i++) {
                //Generating a random number which will be fed into the trainPerceptron function. This will be used in shuffling the data.
                count=count+rand.nextInt(trainExamples.getNumberOfExamples());
                trainPerceptron(trainExamples, inputs, weights, count);

                List<String> trainPredictions = predictPerceptron(trainExamples,inputs,weights);
                double trainSetAccuracy = computeAccuracyPercentage(trainExamples,trainPredictions);

                List<String> tunePredictions = predictPerceptron(tuneExamples,inputs,weights);
                double tuneSetAccuracy = computeAccuracyPercentage(tuneExamples,tunePredictions);

                List<String> testPredictions = predictPerceptron(testExamples,inputs,weights);
                double testSetAccuracy = computeAccuracyPercentage(testExamples,testPredictions);

                //Printing after after 50 epochs and noting the best tune accuracy
                if ((i+1)%50==0 && i!=0) {
                    System.out.printf("Epoch "+(i+1)+": train = ");
                    System.out.printf("%.12f", trainSetAccuracy);
                    System.out.print(" tune = ");
                    System.out.printf("%.12f", tuneSetAccuracy);
                    System.out.print(" test = ");
                    System.out.printf("%.12f", testSetAccuracy);
                    System.out.print('\n');
                    if (maxTuneAccuracy < tuneSetAccuracy) {
                        maxTuneAccuracy = tuneSetAccuracy;
                        testAtMaxTuneAccuracy = testSetAccuracy;
                        maxEpoch = i+1;
                        for (j=0;j<trainExamples.getNumberOfFeatures()+1;j++) {
                            maxTuneInputs[j]=inputs[j];
                            maxTuneWeights[j]=weights[j];
                        }
                    }
                }
            }
            System.out.println("Max Tune Accuracy = "+ maxTuneAccuracy + " at Epoch "+maxEpoch);
            System.out.println("Test set Accuracy at Epoch "+maxEpoch+" = "+testAtMaxTuneAccuracy+". Stopping early to avoid overfitting");

            System.out.println("The test set predictions are: ");
            List<String> maxTestPredictions = predictPerceptron(testExamples,inputs,maxTuneWeights);
            int countCorrect = 0;
            for (j=0;j<testExamples.getNumberOfExamples();j++) {
                //System.out.println("Actual Value: "+testExamples.get(j).getLabel() + '\t' +"Predicted Value: " + maxTestPredictions.get(j));
                System.out.println(maxTestPredictions.get(j));
                if (testExamples.get(j).getLabel().equalsIgnoreCase(maxTestPredictions.get(j))) {
                    countCorrect++;
                }
            }
            //System.out.println("Total correctly predicted: " + countCorrect);

        }

    }
    //This function trains the perceptron using the training set to obtain weights.
    //'count' is a random number passed from main so that we can shuffle the data. We set the learning rate to be 0.1
    private static void trainPerceptron(ListOfExamples trainExamples, double inputs[], double weights[], int count) {
        double sum = 0;
        int i,j;

        inputs[trainExamples.getNumberOfFeatures()] = -1;

        for(i=0;i<trainExamples.getNumberOfExamples();i++) {
            sum=0;
            for (j=0;j<trainExamples.getNumberOfFeatures();j++) {
                if ((trainExamples.get((i+count)%trainExamples.getNumberOfExamples()).get(j)).equals(trainExamples.features[j].getFirstValue())) {
                    inputs[j] = 1;
                } else {
                    inputs [j] = 0;
                }
                sum+=inputs[j]*weights[j];
            }
            sum+=inputs[j]*weights[j];

            for (j=0;j<trainExamples.getNumberOfFeatures()+1;j++) {
                if ((sum>=0) && (trainExamples.get((i+count)%trainExamples.getNumberOfExamples()).label.equals(trainExamples.outputLabel.getFirstValue()))) {
                    weights[j]-=0.1*inputs[j];
                }
                else if ((sum<0) && (trainExamples.get((i+count)%trainExamples.getNumberOfExamples()).label.equals(trainExamples.outputLabel.getSecondValue()))) {
                    weights[j]+=0.1*inputs[j];
                }
            }
        }
    }

    //This function outputs the predictions of the learned model
    private static List<String> predictPerceptron(ListOfExamples Examples, double[] inputs, double[] weights) {
        double sum = 0;
        List<String> predictions = new ArrayList<String>();
        int i,j;
        inputs[Examples.getNumberOfFeatures()] = -1;
        for(i=0;i<Examples.getNumberOfExamples();i++) {
            sum=0;
            for (j=0;j<Examples.getNumberOfFeatures();j++) {
                if ((Examples.get(i).get(j)).equals(Examples.features[j].getFirstValue())) {
                    inputs[j] = 1;
                }
                else {
                    inputs [j] = 0;
                }
                sum+=inputs[j]*weights[j];
            }
            sum+=inputs[j]*weights[j];
            if (sum >= 0) {
                predictions.add(Examples.outputLabel.getSecondValue());
            }
            else {
                predictions.add(Examples.outputLabel.getFirstValue());
            }
        }
        return predictions;
    }

    //This function computes the accuracy percentage depending on what the model returns and what the actual answer is.
    public static double computeAccuracyPercentage(ListOfExamples examples, List<String> predictions) {
        int correctlyClassified = 0;
        int totalExamples = examples.getNumberOfExamples();
        double percentCorrect;
        for (int i=0;i<totalExamples;i++) {
            if (examples.get(i).label.equals(predictions.get(i))) {
                correctlyClassified++;
            }
        }
        percentCorrect = ((double)correctlyClassified/(double)totalExamples) * 100;
        return percentCorrect;
    }
}

// This class, an extension of ArrayList, holds an individual example.
// The new method PrintFeatures() can be used to
// display the contents of the example.
// The items in the ArrayList are the feature values.
class Example extends ArrayList<String> {
    // The name of this example.
    private String name;

    // The output label of this example.
    public String label;

    // The data set in which this is one example.
    private ListOfExamples parent;

    // Constructor which stores the dataset which the example belongs to.
    public Example(ListOfExamples parent) {
        this.parent = parent;
    }

    // Print out this example in human-readable form.
    public void PrintFeatures() {
        System.out.print("Example " + name + ",  label = " + label + "\n");
        for (int i = 0; i < parent.getNumberOfFeatures(); i++) {
            System.out.print("     " + parent.getFeatureName(i)
                    + " = " +  this.get(i) + "\n");
        }
    }

    // Adds a feature value to the example.
    public void addFeatureValue(String value) {
        this.add(value);
    }

    // Accessor methods.
    public String getName() {
        return name;
    }

    public String getLabel() {
        return label;
    }

    // Mutator methods.
    public void setName(String name) {
        this.name = name;
    }

    public void setLabel(String label) {
        this.label = label;
    }
}

/* This class holds all of our examples from one dataset
   (train OR test, not BOTH).  It extends the ArrayList class.
   Be sure you're not confused.  We're using TWO types of ArrayLists.
   An Example is an ArrayList of feature values, while a ListOfExamples is
   an ArrayList of examples. Also, there is one ListOfExamples for the
   TRAINING SET and one for the TESTING SET.
*/
class ListOfExamples extends ArrayList<Example> {
    // The name of the dataset.
    private String nameOfDataset = "";

    // The number of features per example in the dataset.
    private int numFeatures = -1;

    // An array of the parsed features in the data.
    public BinaryFeature[] features;

    // A binary feature representing the output label of the dataset.
    public BinaryFeature outputLabel;

    // The number of examples in the dataset.
    private int numExamples = -1;

    public ListOfExamples() {}

    // Print out a high-level description of the dataset including its features.
    public void DescribeDataset() {
        System.out.println("Dataset '" + nameOfDataset + "' contains "
                + numExamples + " examples, each with "
                + numFeatures + " features.");
        System.out.println("Valid category labels: "
                + outputLabel.getFirstValue() + ", "
                + outputLabel.getSecondValue());
        System.out.println("The feature names (with their possible values) are:");
        for (int i = 0; i < numFeatures; i++)
        {
            BinaryFeature f = features[i];
            System.out.println("   " + f.getName() + " (" + f.getFirstValue() +
                    " or " + f.getSecondValue() + ")");
        }
        System.out.println();
    }

    // Print out ALL the examples.
    public void PrintAllExamples()
    {
        System.out.println("List of Examples\n================");
        for (int i = 0; i < size(); i++)
        {
            Example thisExample = this.get(i);
            thisExample.PrintFeatures();
        }
    }

    // Print out the SPECIFIED example.
    public void PrintThisExample(int i)
    {
        Example thisExample = this.get(i);
        thisExample.PrintFeatures();
    }

    // Returns the number of features in the data.
    public int getNumberOfFeatures() {
        return numFeatures;
    }
    // Returns the number of features in the data.
    public int getNumberOfExamples() {
        return numExamples;
    }

    // Returns the name of the ith feature.
    public String getFeatureName(int i) {
        return features[i].getName();
    }

    // Takes the name of an input file and attempts to open it for parsing.
    // If it is successful, it reads the dataset into its internal structures.
    // Returns true if the read was successful.
    public boolean ReadInExamplesFromFile(String dataFile) {
        nameOfDataset = dataFile;

        // Try creating a scanner to read the input file.
        Scanner fileScanner = null;
        try {
            fileScanner = new Scanner(new File(dataFile));
        } catch(FileNotFoundException e) {
            return false;
        }

        // If the file was successfully opened, read the file
        this.parse(fileScanner);
        return true;
    }

    /**
     * Does the actual parsing work. We assume that the file is in proper format.
     *
     * @param fileScanner a Scanner which has been successfully opened to read
     * the dataset file
     */
    public void parse(Scanner fileScanner) {
        // Read the number of features per example.
        numFeatures = Integer.parseInt(parseSingleToken(fileScanner));

        // Parse the features from the file.
        parseFeatures(fileScanner);

        // Read the two possible output label values.
        String labelName = "output";
        String firstValue = parseSingleToken(fileScanner);
        String secondValue = parseSingleToken(fileScanner);
        outputLabel = new BinaryFeature(labelName, firstValue, secondValue);

        // Read the number of examples from the file.
        numExamples = Integer.parseInt(parseSingleToken(fileScanner));

        parseExamples(fileScanner);
    }

    /**
     * Returns the first token encountered on a significant line in the file.
     *
     * @param fileScanner a Scanner used to read the file.
     */
    private String parseSingleToken(Scanner fileScanner) {
        String line = findSignificantLine(fileScanner);

        // Once we find a significant line, parse the first token on the
        // line and return it.
        Scanner lineScanner = new Scanner(line);
        return lineScanner.next();
    }

    /**
     * Reads in the feature metadata from the file.
     *
     * @param fileScanner a Scanner used to read the file.
     */
    private void parseFeatures(Scanner fileScanner) {
        // Initialize the array of features to fill.
        features = new BinaryFeature[numFeatures];

        for(int i = 0; i < numFeatures; i++) {
            String line = findSignificantLine(fileScanner);

            // Once we find a significant line, read the feature description
            // from it.
            Scanner lineScanner = new Scanner(line);
            String name = lineScanner.next();
            String dash = lineScanner.next();  // Skip the dash in the file.
            String firstValue = lineScanner.next();
            String secondValue = lineScanner.next();
            features[i] = new BinaryFeature(name, firstValue, secondValue);
        }
    }

    private void parseExamples(Scanner fileScanner) {
        // Parse the expected number of examples.
        for(int i = 0; i < numExamples; i++) {
            String line = findSignificantLine(fileScanner);
            Scanner lineScanner = new Scanner(line);

            // Parse a new example from the file.
            Example ex = new Example(this);

            String name = lineScanner.next();
            ex.setName(name);

            String label = lineScanner.next();
            ex.setLabel(label);

            // Iterate through the features and increment the count for any feature
            // that has the first possible value.
            for(int j = 0; j < numFeatures; j++) {
                String feature = lineScanner.next();
                ex.addFeatureValue(feature);
            }

            // Add this example to the list.
            this.add(ex);
        }
    }

    /**
     * Returns the next line in the file which is significant (i.e. is not
     * all whitespace or a comment.
     *
     * @param fileScanner a Scanner used to read the file
     */
    private String findSignificantLine(Scanner fileScanner) {
        // Keep scanning lines until we find a significant one.
        while(fileScanner.hasNextLine()) {
            String line = fileScanner.nextLine().trim();
            if (isLineSignificant(line)) {
                return line;
            }
        }

        // If the file is in proper format, this should never happen.
        System.err.println("Unexpected problem in findSignificantLine.");

        return null;
    }

    /**
     * Returns whether the given line is significant (i.e., not blank or a
     * comment). The line should be trimmed before calling this.
     *
     * @param line the line to check
     */
    private boolean isLineSignificant(String line) {
        // Blank lines are not significant.
        if(line.length() == 0) {
            return false;
        }

        // Lines which have consecutive forward slashes as their first two
        // characters are comments and are not significant.
        if(line.length() > 2 && line.substring(0,2).equals("//")) {
            return false;
        }

        return true;
    }
}

/**
 * Represents a single binary feature with two String values.
 */
class BinaryFeature {
    private String name;
    private String firstValue;
    private String secondValue;

    public BinaryFeature(String name, String first, String second) {
        this.name = name;
        firstValue = first;
        secondValue = second;
    }

    public String getName() {
        return name;
    }

    public String getFirstValue() {
        return firstValue;
    }

    public String getSecondValue() {
        return secondValue;
    }
}

class Utilities
{
    // This method can be used to wait until you're ready to proceed.
    public static void waitHere(String msg)
    {
        System.out.print("\n" + msg);
        try { System.in.read(); }
        catch(Exception e) {} // Ignore any errors while reading.
    }
}
