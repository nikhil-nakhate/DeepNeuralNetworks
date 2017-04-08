/**
 * Lab2.java
 * February 2017
 * Authors:
 * 		- Felipe Gutierrez Barragan
 * 		- Nikhil Nakhate
 * Program Description: Single layer neural network program
 */

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;
import java.util.Random;

public class Lab2 {
    public static void main(String[] args) {
        System.out.println(args.length);
        // if (args.length != 4) {
        // 	System.err.println("Please provide a data file on the " + "command line: java Lab2" + " <data_filename> <alpha> <num_hu> <max_epoch>");
        // 	System.exit(1);
        // }
//        if (args.length != 1) {
//            System.err.println("Please provide a data file on the " + "command line: java Lab2" + " <data_filename>");
//            System.exit(1);
//        }


        // Perform I/O of all the datasets
        String data_filename = "src/protein-secondary-structure.data";
        // double alpha =  Double.parseDouble(args[1]);
        // int num_hu = Integer.parseInt(args[2]);
        // int max_epoch = Integer.parseInt(args[3]);
        // double beta =  Double.parseDouble(args[4]);
        // double lamda =  Double.parseDouble(args[5]);
        double alpha = 0.01;
        int num_hu = 500;
        int max_epoch = 1000;
        double beta =  0.9;
        double lamda =  0.01;

        int tune_period = 10;
        int num_features = 21 * 17;
        int num_labels = 3;

        System.out.println("Using the following input files: \n" + data_filename);
        System.out.println("Using the following parameters:");
        System.out.println("alpha:" + alpha);
        System.out.println("num_hu:" + num_hu);
        System.out.println("max_epoch:" + max_epoch);
        System.out.println("tune_period:" + tune_period);

        ProteinDataset dataset = new ProteinDataset();
        dataset.parseProteinFile(data_filename);
        dataset.printDatasetStats();



        NeuralNet nn = new NeuralNet(alpha, max_epoch, num_features, num_labels, num_hu, tune_period, beta, lamda);

        nn.train(dataset);
        double test_accuracy = nn.test(dataset.proteins_test,true);
        System.out.println("Final Test Accuracy: " + test_accuracy);


        String file_prefix = "ReLUMomentumResults";
        nn.writeResults(file_prefix);

    }
}