import java.util.ArrayList;
import java.util.List;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

/**
 * A protein object.
 *
 */
public class ProteinDataset {
	
	/**
	 * List of amino acid characters
	 */
	public List<Protein> proteins_train;
	public List<Protein> proteins_tune;
	public List<Protein> proteins_test;
	
	/**
	 * The label of the instance.
	 */
	public int num_proteins;

	public char [] possible_labels = {'_','e','h'};
	
	
	/**
	 * Constructor.
	 */
	public ProteinDataset() {
		proteins_train = new ArrayList<Protein> ();
		proteins_tune = new ArrayList<Protein> ();
		proteins_test = new ArrayList<Protein> ();
	}


	public void parseProteinFile(String filename) {
		
		// Try creating a scanner to read the input file.
		Scanner file_scanner = null;
		try {
			file_scanner = new Scanner(new File(filename));
		} catch(FileNotFoundException e) {
			System.err.println("Could not find file '" + filename + "'.");
			System.exit(1);
		}
		int counter = 1;

		String line = findSignificantLine(file_scanner);
		while(file_scanner.hasNextLine()) {
			line = findSignificantLine(file_scanner);
			Protein p = parseProtein(file_scanner, line);
			p.generateOneHotFeatureArray();	
			p.generateLabels();
			if(counter % 5 == 0) { 
				proteins_tune.add(p);
			}
			else if (counter % 5 == 1){
				proteins_test.add(p);
			}
			else {
				proteins_train.add(p);
			}
			counter++;
		}

		// What is left are the examples. So we iterate and add them		

	}

	private Protein parseProtein(Scanner file_scanner, String firstLine) {

		Protein p = new Protein();
		if(!isProteinSeparator(firstLine)) {
			p.addAA(firstLine.charAt(0));
			p.addAALabel(firstLine.charAt(2));
		}

		boolean done = false;
		while(file_scanner.hasNextLine() && !done) {   
			String line = file_scanner.nextLine().trim();
		    // System.out.println(isProteinSeparator(line));
		    if(isProteinSeparator(line)) {
				// addInstance(line);
				// System.out.println("Protein end indicator: " + line);
				done = true;
		    }
		    else if(isLineSignificant(line)) {
		    	p.addAA(line.charAt(0));
		    	p.addAALabel(line.charAt(2));
		    }
		}
		// System.out.println("aa.size = " + p.aa.size());
		// System.out.println("aa_labels.size = " + p.aa_labels.size());
		return p;
	}

  /**
   * Returns the next line in the file which is significant (i.e. is not
   * all whitespace or a comment.
   *
   * @param file_scanner a Scanner used to read the file
   */
	private String findSignificantLine(Scanner file_scanner) {
		// Keep scanning lines until we find a significant one.
		while(file_scanner.hasNextLine()) {
			String line = file_scanner.nextLine().trim();
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
		// Lines which have # signs are not significant
		// characters are comments and are not significant.
		if(line.length() > 0 && line.substring(0,1).equals("#")) {
			return false;
		}
    	return true;
	}

	/**
	 * Returns whether the given line is the end of protein.
	 *
	 */
	private boolean isProteinSeparator(String line) {
		// System.out.println(line);
		if(line.length() > 1 && line.substring(0,2).equals("<>")) {
			return true;
		}
		if(line.length() > 2 && line.substring(0,3).equals("end")) {
			return true;
		}
		if(line.length() > 4 && line.substring(0,5).equals("<end>")) {
			return true;
		}

    	return false;
	}

	public void printDatasetStats() {
		int totalAA_train = 0;
		int totalAA_tune = 0;
		int totalAA_test = 0;
		for(int i = 0;i < proteins_train.size();i++) {
			totalAA_train += proteins_train.get(i).aa.size(); 
		}
		for(int i = 0;i < proteins_tune.size();i++) {
			totalAA_tune += proteins_tune.get(i).aa.size(); 
		}
		for(int i = 0;i < proteins_test.size();i++) {
			totalAA_test += proteins_test.get(i).aa.size(); 
		}
		int totalAA = totalAA_train + totalAA_tune + totalAA_test;
		int totalProteins = proteins_train.size() + proteins_tune.size() + proteins_test.size();

		System.out.println("Total number of train proteins = " + proteins_train.size());
		System.out.println("Total number of train AA = " + totalAA_train);
		System.out.println("Total number of tune proteins = " + proteins_tune.size());
		System.out.println("Total number of tune AA = " + totalAA_tune);
		System.out.println("Total number of test proteins = " + proteins_test.size());
		System.out.println("Total number of test AA = " + totalAA_test);

		System.out.println("Total number of proteins = " + totalProteins);
		System.out.println("Total number of AA = " + totalAA);
	}

	
}