import java.util.ArrayList;
import java.util.List;

/**
 * A protein object.
 *
 */
public class Protein {
	
	/**
	 * List of amino acid characters
	 */
	public List<Character> aa;
	public List<Character> aa_labels;
	public double[] features;
	public double[][] labels;
	
	int padding;
	
	/**
	 * Constructor.
	 */
	public Protein() {
		this.padding = 8;
		aa = new ArrayList<Character> ();
		aa_labels = new ArrayList<Character> ();
		addSlidingWindowPadding(this.padding);
	}

	/**
	 */
	public void addAA(Character amino_acid) {
		aa.add(amino_acid);
	}
	
	/**
	 */
	public void addAALabel(Character aa_label) {
		aa_labels.add(aa_label);
	}

	public void addSlidingWindowPadding(int length) {
		for(int i = 0;i < length;i++) {
			aa.add(' ');
			aa_labels.add(' ');
		}
	}

	public void generateOneHotFeatureArray() {
		addSlidingWindowPadding(this.padding);
		// System.out.println("Generating feature array of length: " + 21 + "x" + aa.size() + "=" + 21*aa.size());
		features = new double[21 * aa.size()];
		int currIndex = 0;
		for(int i = 0;i < aa.size();i++) {
			// System.out.println("oneh_f for: " + aa.get(i));
			double[] oneh_f = getOneHotFeature(aa.get(i));
			for(int j = 0;j < 21;j++) {
				currIndex = i*21 + j;
				// System.out.println(currIndex);
				features[currIndex] = oneh_f[j];
			}
			
			// for(int j = 0; j < 21;j++) {
			// 	System.out.print(oneh_f[j]+", ");				
			// }
			// System.out.println("");
		}
	}

	public void generateLabels() {
		labels = new double[aa_labels.size() - 16][3];
		int index = 0;
		int counter = 0;
		for(int i = 8;i < aa_labels.size() - 8;i++) {
			index = checkLabelIndex(aa_labels.get(i));
			// if(index >= 0) {
			labels[counter][index] = 1.;
			// }
			counter++;
		}
	}


	public int checkLabelIndex(Character l) {
		switch(l) {
			case ' ' :
				return -1;
			case '_' :
				return 0;
			case 'e' :
				return 1;
			case 'h' :
				return 2;
			default :
				System.out.println("Incorrect output character: " + l);	
				System.exit(1);
		}
		
		return -1;
	}

	private double[] getOneHotFeature(Character aa) {
		double [] oneh_aa = new double[21]; 

		switch(aa) {
			case ' ' :
				oneh_aa[0] = 1.0;
		    	break; 
			case 'A' :
				oneh_aa[1] = 1.0;
		    	break; 
			case 'C' :
				oneh_aa[2] = 1.0;
		    	break; 
			case 'D' :
				oneh_aa[3] = 1.0;
		    	break; 
			case 'E' :
				oneh_aa[4] = 1.0;
		    	break; 
			case 'F' :
				oneh_aa[5] = 1.0;
		    	break; 
			case 'G' :
				oneh_aa[6] = 1.0;
		    	break; 
			case 'H' :
				oneh_aa[7] = 1.0;
		    	break; 
			case 'I' :
				oneh_aa[8] = 1.0;
		    	break; 
			case 'K' :
				oneh_aa[9] = 1.0;
		    	break; 
			case 'L' :
				oneh_aa[10] = 1.0;
		    	break; 
			case 'M' :
				oneh_aa[11] = 1.0;
		    	break; 
			case 'N' :
				oneh_aa[12] = 1.0;
		    	break; 
			case 'P' :
				oneh_aa[13] = 1.0;
		    	break; 
			case 'Q' :
				oneh_aa[14] = 1.0;
		    	break; 
			case 'R' :
				oneh_aa[15] = 1.0;
		    	break; 
			case 'S' :
				oneh_aa[16] = 1.0;
		    	break; 
			case 'T' :
				oneh_aa[17] = 1.0;
		    	break; 
			case 'V' :
				oneh_aa[18] = 1.0;
		    	break; 
			case 'W' :
				oneh_aa[19] = 1.0;
		    	break; 
			case 'Y' :
				oneh_aa[20] = 1.0;
		    	break; 
		   		   
		   	// You can have any number of case statements.
		   	default : 
				System.err.println("Incorrect input character for one hot encoding: " + aa);
				System.exit(1);

		}


		return oneh_aa;
	}

}