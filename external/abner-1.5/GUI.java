/* ****************************************************************

   Copyright (C) 2004 Burr Settles, University of Wisconsin-Madison,
   Dept. of Computer Sciences and Dept. of Biostatistics and Medical
   Informatics.

   This file is part of the "ABNER (A Biomedical Named Entity
   Recognizer)" system. It requires Java 1.4. This software is
   provided "as is," and the author makes no representations or
   warranties, express or implied. For details, see the "README" file
   included in this distribution.

   This software is provided under the terms of the Common Public
   License, v1.0, as published by http://www.opensource.org. For more
   information, see the "LICENSE" file included in this distribution.

   **************************************************************** */

/**
   "GUI.java" implements the interactive, graphical user interface for
   the ABNER system.

   @author Burr Settles <a href="http://www.cs.wisc.edu/~bsettles">bsettles&#64;&#99;s&#46;&#119;i&#115;&#99;&#46;&#101;d&#117;</a> 
   @version 1.5 (March 2005)
*/

import abner.*;

import javax.swing.*;
import javax.swing.text.*;
import java.awt.*;
import java.awt.event.*;
import java.net.URL;
import java.io.*;
import java.lang.*;
import java.net.*;
import java.util.Vector;
import java.util.Iterator;
import java.util.List;
import java.util.Collections;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.regex.*;

import edu.umass.cs.mallet.base.fst.*;
import edu.umass.cs.mallet.base.minimize.*;
import edu.umass.cs.mallet.base.pipe.*;
import edu.umass.cs.mallet.base.pipe.iterator.*;
import edu.umass.cs.mallet.base.types.*;

public class GUI extends JFrame implements ActionListener, ItemListener {

    //****************
    // global data members

    // demo strings
    static String demo1 = "Analysis of myeloid-associated genes in human hematopoietic progenitor cells.\nBello-Fernandez et al. Exp Hematol.  1997 Oct;25(11):1158-66.\n\nThe distribution of myeloid lineage-associated cytokine receptors and lysosomal proteins was analyzed in human CD34+ cord blood cell (CB) subsets at different stages of myeloid commitment by reverse-transcriptase polymerase chain reaction (RT-PCR). The highly specific granulomonocyte-associated lysosomal proteins myeloperoxidase (MPO) and lysozyme (LZ), as well as the transcription factor PU.1, were already detectable in the most immature CD34+Thy-1+ subset. Messenger RNA (mRNA) levels for the granulocyte-colony stimulating factor (G-CSF) receptor, granulocyte-macrophage (GM)-CSF receptor alpha subunit and tumor necrosis factor (TNF) receptors I (p55) and II (p75) were also detected in this subset in addition to c-kit and flt-3, receptors known to be expressed on progenitor cells. By contrast, the monocyte-macrophage colony stimulating factor (M-CSF) receptor was largely absent at this stage and in the CD34+Thy-1-CD45RA- subsets. The M-CSF receptor was first detectable in the myeloid-committed CD34+Thy-l-CD45RA+ subset. All other molecules studied were found to be expressed at this stage of differentiation. Different cocktails of the identified ligands were added to sorted CD34+Thy-1+ single cells. Low proliferative capacity was observed after 1 week in culture in the presence of stem cell factor (SCF) + Flt-3 ligand (FL) + G-CSF. Addition of GM-CSF to this basic cocktail consistently increased the clonogenic capacity of single CD34+Thy-1+ cells, and this effect was further enhanced (up to 72.3 +/- 4.3% on day 7) by the inclusion of TNF-alpha. In conclusion, the presence of myeloid-associated growth factor receptor transcripts in CD34+ CB subsets does not discriminate the various stages of differentiation, with the exception of the M-CSF receptor. In addition, we show that TNF-alpha is a potent costimulatory factor of the very immature CD34+Thy-1+ CB subset.";
    static String demo2 = "AP-1 (Fos/Jun) transcription factors in hematopoietic differentiation and apoptosis.\nLiebermann et al. Int J Oncol.  1998 Mar;12(3):685-700.\n\nA combination of in vitro and in vivo molecular genetic approaches have provided evidence to suggest that AP-1 (Fos/Jun) transcription factors play multiple roles in functional development of hematopoietic precursor cells into mature blood cells along most, if not all, of the hematopoietic cell lineages. This includes the monocyte/macrophage, granulocyte, megakaryocyte, mastocyte and erythroid lineages. In addition, studies using c-Fos knockout mice have established a unique role for Fos, as a member of the AP-1 transcription factor complex, in determining the differentiation and activity of progenitors of the osteoclast lineage, a population of bone-forming cells which are of hematopoietic origin as well. Evidence has also accumulated to implicate AP-1 (Fos/Jun) transcription factor complexes as both positive and negative modulators of distinct apoptotic pathways in many cell types, including cells of hematopoietic origin. Fos/Jun have been implicated as positive modulators of apoptosis induced in hematopoietic progenitor cells of the myeloid lineage, a function that may relate to the control of blood cell homeostasis, as well as in programmed cell death associated with terminal differentiation of many other cell types, and apoptosis associated with withdrawal of growth/survival factors. On the other hand, the study of apoptosis induced in mammalian cells has implicated AP-1 in the protection against apoptosis induced by DNA-damaging agents. However, evidence to the contrary has been obtained as well, suggesting that AP-1 may function to modulate stress-induced apoptosis either positively or negatively, depending on the microenvironment and the cell type in which the stress stimulus is induced.";
    static String demo3 = "Heterozygous PU.1 mutations are associated with acute myeloid leukemia.\nMueller et al. Blood. 2002 Aug 1;100(3):998-1007.\n\nThe transcription factor PU.1 is required for normal blood cell development. PU.1 regulates the expression of a number of crucial myeloid genes, such as the macrophage colony-stimulating factor (M-CSF) receptor, the granulocyte colony-stimulating factor (G-CSF) receptor, and the granulocyte-macrophage colony-stimulating factor (GM-CSF) receptor. Myeloid cells derived from PU.1(-/-) mice are blocked at the earliest stage of myeloid differentiation, similar to the blast cells that are the hallmark of human acute myeloid leukemia (AML). These facts led us to hypothesize that molecular abnormalities involving the PU.1 gene could contribute to the development of AML. We identified 10 mutant alleles of the PU.1 gene in 9 of 126 AML patients. The PU.1 mutations comprised 5 deletions affecting the DNA-binding domain, and 5 point mutations in 1) the DNA-binding domain (2 patients), 2) the PEST domain (2 patients), and 3) the transactivation domain (one patient). DNA binding to and transactivation of the M-CSF receptor promoter, a direct PU.1 target gene, were deficient in the 7 PU.1 mutants that affected the DNA-binding domain. In addition, these mutations decreased the ability of PU.1 to synergize with PU.1-interacting proteins such as AML1 or c-Jun in the activation of PU.1 target genes. This is the first report of mutations in the PU.1 gene in human neoplasia and suggests that disruption of PU.1 function contributes to the block in differentiation found in AML patients.";

    // source text
    static JTextArea sourceText = new JTextArea("");
    static JScrollPane sourcePane;
    // labeled text
    static JTextPane labelText = new JTextPane();
    static JScrollPane labelPane;	
    static JPanel toolPane = new JPanel();

    // very important: the CRF itself and its feature pipes!!
    static Tagger taggerNLPBA;
    static Tagger taggerBIOCR;
    static Tagger tagger;

    ////////////////////////////////////////////////////////////////
    // builds the source text panel
    public void updateSource() {
        sourceText.setFont(new Font("Sans-Serif", Font.PLAIN, 11));
	sourceText.setLineWrap(true);
	sourceText.setWrapStyleWord(true);
	sourcePane = new JScrollPane(sourceText);
        sourcePane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
        sourcePane.setPreferredSize(new Dimension(550, 200));
        sourcePane.setBorder(BorderFactory.createCompoundBorder(BorderFactory.createCompoundBorder(BorderFactory.createTitledBorder("Source Text"),
												   BorderFactory.createEmptyBorder(5,5,5,5)),
								sourcePane.getBorder()));
    }

    ////////////////////////////////////////////////////////////////
    // builds the labeled text panel
    public void updateLabel() {
        labelText.setFont(new Font("Sans-Serif", Font.PLAIN, 11));
	labelText.setEditable(false);
	labelPane = new JScrollPane(labelText);
        StyledDocument doc = labelText.getStyledDocument();
	addStylesToDocument(doc);
        labelPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
        labelPane.setPreferredSize(new Dimension(550, 200));
        labelPane.setBorder(BorderFactory.createCompoundBorder(BorderFactory.createTitledBorder("Annotated Text"),
							       BorderFactory.createEmptyBorder(5,5,5,5)));
    }

    ////////////////////////////////////////////////////////////////
    // generic "make the GUI"
    public JPanel createContent() {
	JPanel contentPane = new JPanel(new BorderLayout());
	// build the main text panels
	updateSource();
	updateLabel();
	// join then im a split panel
        JSplitPane splitPane = new JSplitPane(JSplitPane.VERTICAL_SPLIT,
                                              sourcePane,
                                              labelPane);
        splitPane.setOneTouchExpandable(true);
        splitPane.setResizeWeight(0.5);
	// labeling tools @ bottom...
	JButton labelButton = new JButton("Annotate!");
	labelButton.addActionListener(this);
 	toolPane.add(labelButton);
	toolPane.add(new JLabel("<html><font style='background-color:#ffff99;'>protein</font></html>"));
	toolPane.add(new JLabel("<html><font style='background-color:#99ff99;'>DNA</font></html>"));
	toolPane.add(new JLabel("<html><font style='background-color:#99ffff;'>RNA</font></html>"));
	toolPane.add(new JLabel("<html><font style='background-color:#ffccff;'>cell line</font></html>"));
	toolPane.add(new JLabel("<html><font style='background-color:#ffcccc;'>cell type</font></html>"));
        toolPane.setBorder(BorderFactory.createCompoundBorder(
							      BorderFactory.createTitledBorder("Entity Recognition Tools"),
							      BorderFactory.createEmptyBorder(1,1,1,1)));
	// put it all together
	contentPane.add(splitPane, BorderLayout.CENTER);
	contentPane.add(toolPane, BorderLayout.SOUTH);
	return contentPane;
    }

    ////////////////////////////////////////////////////////////////
    // listen to the menu selections and act accordingly.
    public void actionPerformed(ActionEvent e) {
	boolean status = false;
	String cmd = e.getActionCommand();
	try {
	    // debugging purposes
	    //	    System.err.println("EVENT: ["+cmd+"]");
	    StyledDocument doc = labelText.getStyledDocument();
	    // "new" (clear source text)
	    if (cmd.equals("New")) {
		sourceText.setText("");
		doc.remove(0,doc.getLength());
	    }
	    // "Open" (new file)
	    else if (cmd.equals("Open")) {
		status = openFile();
		if(!status)
		    JOptionPane.showMessageDialog(null, "Error opening file!", 
						  "File Open Error", 
						  JOptionPane.ERROR_MESSAGE);
		doc.remove(0,doc.getLength());
		sourceText.setCaretPosition(0);
	    }
	    // "Save" (current labeled text)
	    else if(cmd.equals("Save")) {
		status = saveFile();
		if( !status)
		    JOptionPane.showMessageDialog(null, "IO error in saving file!!", 
						  "File Save Error",
						  JOptionPane.ERROR_MESSAGE);
	    }
	    // quit program
	    else if (cmd.equals("Quit"))
		System.exit(1);
	    // examples...
	    else if (cmd.startsWith("Demo")) {
		doc.remove(0,doc.getLength());
		switch(cmd.charAt(cmd.length()-1)) {
		case '1': sourceText.setText(demo1); 
		    sourceText.setCaretPosition(0); break;
		case '2': sourceText.setText(demo2);
		    sourceText.setCaretPosition(0); break;
		case '3': sourceText.setText(demo3);
		    sourceText.setCaretPosition(0); break;
		}
	    }
	    // display about information
	    else if (cmd.startsWith("About")) {
		AboutDialog a = new AboutDialog(this);
	    }
	    // batch annotation
	    else if (cmd.startsWith("Batch")) {
		batchAnnotate();
	    }
	    // take text from "source", tag it, put it in "label"
	    else if (cmd.startsWith("Annotate")) {
		labelTheText();
	    }
	    // toggle the active model...
	    else if (cmd.startsWith("NLPBA")) {
		tagger = taggerNLPBA;
	    }
	    else if (cmd.startsWith("BioCreative")) {
		tagger = taggerBIOCR;
	    }
	    // set tokenization
	    else if (cmd.startsWith("Auto-Tok")) {
		taggerNLPBA.setTokenization(!taggerNLPBA.getTokenization());
		taggerBIOCR.setTokenization(!taggerBIOCR.getTokenization());
	    }

	} catch (BadLocationException ble) {
	    System.err.println("Exception trying to reset 'labelText'");
	} catch (Exception ex) {
	    System.err.println("Some file not found.");
	}
    }

    ////////////////////////////////////////////////////////////////
    // runs the batch annotation functionality...
    void batchAnnotate() throws java.io.FileNotFoundException, java.io.IOException{
	// talk to me
	System.err.println("[Begin batch annotation.]");
	// get the dir + format...
	BatchConverter bc = new BatchConverter(this);
	String[] bla = bc.getInfo();
	List filesToAnnotate = getFileListing(new File(bla[0]));
	Iterator filesIter = filesToAnnotate.iterator();
	while( filesIter.hasNext() ){
	    File f = (File)filesIter.next();
	    // make sure it's not something that we wrote!
	    if (!f.toString().endsWith(".iob") &&
		!f.toString().endsWith(".abner") &&
		!f.toString().endsWith(".sgml")) {
		File newFile = new File(f.toString()+"."+bla[1]);
		String fileString = readFile(f);
		if(fileString != null) {
		    if (bla[1].equals("sgml"))
			writeFile(newFile, tagger.tagSGML(fileString));
		    if (bla[1].equals("iob"))
			writeFile(newFile, tagger.tagIOB(fileString));
		    if (bla[1].equals("abner"))
			writeFile(newFile, tagger.tagABNER(fileString));
		    // report done.
		    System.err.println("Annotated file: '"+newFile.getPath()+"'");
		}
	    }
	}
	// talk again...
	System.err.println("[Batch annotation complete!]");
    }

    ////////////////////////////////////////////////////////////////
    // Recursively walk a directory tree and return a List of all
    // Files found; the List is sorted using File.compareTo.
    static public List getFileListing( File aStartingDir ) throws FileNotFoundException{
	validateDirectory(aStartingDir);
	List result = new ArrayList();
	// do it
	File[] filesAndDirs = aStartingDir.listFiles();
	List filesDirs = Arrays.asList(filesAndDirs);
	Iterator filesIter = filesDirs.iterator();
	File file = null;
	while ( filesIter.hasNext() ) {
	    file = (File)filesIter.next();
	    result.add(file); //always add, even if directory
	    if (!file.isFile()) {
		//must be a directory
		//recursive call!
		List deeperList = getFileListing(file);
		result.addAll(deeperList);
	    }
	}
	Collections.sort(result);
	return result;
    }

    ////////////////////////////////////////////////////////////////
    // Directory is valid if it exists, does not represent a file, and can be read.
    static private void validateDirectory (File aDirectory) throws FileNotFoundException {
	if (aDirectory == null) {
	    throw new IllegalArgumentException("Directory should not be null.");
	}
	if (!aDirectory.exists()) {
	    throw new FileNotFoundException("Directory does not exist: " + aDirectory);
	}
	if (!aDirectory.isDirectory()) {
	    throw new IllegalArgumentException("Is not a directory: " + aDirectory);
	}
	if (!aDirectory.canRead()) {
	    throw new IllegalArgumentException("Directory cannot be read: " + aDirectory);
	}
    }

    ////////////////////////////////////////////////////////////////
    // opens text files
    boolean openFile() {
	File file = null;
	JFileChooser fc = new JFileChooser();
	fc.setDialogTitle("Open File");
	// Choose only files, not directories
	fc.setFileSelectionMode(JFileChooser.FILES_ONLY);
	// Start in current directory
	fc.setCurrentDirectory(new File("."));
	// Set filter for .txt source files.
	fc.setFileFilter(new AbnerFilter(".txt", "Text"));
	// Now open chooser
	int result = fc.showOpenDialog(this);
	if( result == JFileChooser.CANCEL_OPTION) {
	    return true;
	}
	else if( result == JFileChooser.APPROVE_OPTION) {
	    file = fc.getSelectedFile();
	    String fileString = readFile(file);
	    if(fileString != null)
		sourceText.setText(fileString);
	    else
		return false;
	}
	else {
	    return false;
	}
	return true;
    }

    ////////////////////////////////////////////////////////////////
    // save dialog: SGML, IOB, and ABNER formats supported
    boolean saveFile() {
	File file = null;
	JFileChooser fc = new JFileChooser();
	// Start in current directory
	fc.setCurrentDirectory(new File("."));
	// Set filter for output files.
	fc.addChoosableFileFilter(new AbnerFilter(".sgml","Generic SGML"));
	fc.addChoosableFileFilter(new AbnerFilter(".iob","CoNLL-style IOB"));
	fc.addChoosableFileFilter(new AbnerFilter(".abner","ABNER training format"));
	// Set to a default name for save.
	fc.setSelectedFile(file);
	// Open chooser dialog
	int result = fc.showSaveDialog(this);
	if( result == JFileChooser.CANCEL_OPTION) {
	    return true;
	}
	else if( result == JFileChooser.APPROVE_OPTION) {
	    file = fc.getSelectedFile();
	    String ext = ((AbnerFilter)fc.getFileFilter()).getExtension();
	    // tack on the extension if not provided
	    if (!file.getName().endsWith(ext)) {
		file = new File(file.getPath()+ext);
	    }
	    // overwrite dialog
	    if( file.exists()) {
		int response = JOptionPane.showConfirmDialog(null,
							     "Overwrite existing file \""+file.getName()+"\"?",
							     "Confirm Overwrite",
							     JOptionPane.OK_CANCEL_OPTION,
							     JOptionPane.QUESTION_MESSAGE);
		if( response == JOptionPane.CANCEL_OPTION) return false;
	    }
	    // write out the appropriate format
	    if (ext.equals(".iob"))
		return writeFile(file, tagger.tagIOB(sourceText.getText()));
	    else if (ext.equals(".sgml"))
		return writeFile(file, tagger.tagSGML(sourceText.getText()));
	    else if (ext.equals(".abner"))
		return writeFile(file, tagger.tagABNER(sourceText.getText()));
	    else {
		return false;
	    }
	}
	else {
	    return false;
	}
    }


    ////////////////////////////////////////////////////////////////
    // actually reads in the chosen text file
    public String readFile(File file) {
	StringBuffer fileBuffer;
	String fileString=null;
	String line;
	try {
	    FileReader in = new FileReader( file);
	    BufferedReader dis =
		new BufferedReader(in);
	    fileBuffer = new StringBuffer () ;
	    while ((line = dis.readLine()) != null) {
		fileBuffer.append(line + "\n");
	    }
	    in.close();
	    fileString = fileBuffer.toString();
	} catch (IOException e ) {
	    return null;
	}
	return fileString;
    }

    ////////////////////////////////////////////////////////////////
    // actually writes out the chosen output file
    public static boolean writeFile(File file, String dataString) {
	try {
	    PrintWriter out =
		new PrintWriter(new BufferedWriter(new FileWriter(file)));
	    out.print(dataString);
	    out.flush();
	    out.close();
	} catch (IOException e) {
	    return false;
	}
	return true;
    }

    // stub routine. just so things compile...
    public void itemStateChanged(ItemEvent e) {}
    

    ////////////////////////////////////////////////////////////////
    // this is what will actually do the labeling...
    private void labelTheText() {
        try {
	    // actually do the labeling bit...
	    Vector res = tagger.getSegments(sourceText.getText());
	    // now do the stylizing...
	    StyledDocument doc = labelText.getStyledDocument();
	    doc.remove(0,doc.getLength());
            for (int i=0; i < res.size(); i++) {
		String[][] sent = (String[][])res.get(i);
		for (int j=0; j<sent[0].length; j++) {
		    doc.insertString(doc.getLength(),
				     sent[0][j],
				     doc.getStyle(sent[1][j]));
		    doc.insertString(doc.getLength(), " ", 
				     doc.getStyle("O"));
		}
		doc.insertString(doc.getLength(), "\n", 
				 doc.getStyle("O"));
            }
	    // and reset to the top
	    labelText.setCaretPosition(0);
        } catch (Exception e) {
            System.err.println(e);
        }
    }



    //################################################################
    //################################################################


    ////////////////////////////////////////////////////////////////
    // define styles for entity highlighting
    protected void addStylesToDocument(StyledDocument doc) {
        //Initialize some styles.
        Style def = StyleContext.getDefaultStyleContext().
	    getStyle(StyleContext.DEFAULT_STYLE);
        Style regular = doc.addStyle("O", def);
	// PROTEIN
        Style s = doc.addStyle("PROTEIN", regular);
        StyleConstants.setBackground(s, new Color(255, 255, 153));
	StyleConstants.setBold(s, true);
	// DNA
	s = doc.addStyle("DNA", regular);
        StyleConstants.setBackground(s, new Color(153, 255, 153));
	StyleConstants.setBold(s, true);
	// RNA
	s = doc.addStyle("RNA", regular);
        StyleConstants.setBackground(s, new Color(153, 255, 255));
	StyleConstants.setBold(s, true);
	// cell line
	s = doc.addStyle("CELL_LINE", regular);
        StyleConstants.setBackground(s, new Color(255, 204, 255));
	StyleConstants.setBold(s, true);
	// cell type
	s = doc.addStyle("CELL_TYPE", regular);
        StyleConstants.setBackground(s, new Color(255, 204, 204));
	StyleConstants.setBold(s, true);
    }

    ////////////////////////////////////////////////////////////////
    // to make that pretty menu....
    private JMenuBar createMenuBar() {
	JMenuBar myMenu = new JMenuBar();
	//////// FILE MENU
	JMenu fileMenu = new JMenu("File");
	fileMenu.setMnemonic('F');
	// new
	JMenuItem item = new JMenuItem("New", 'N');
        item.setAccelerator(KeyStroke.getKeyStroke('N', ActionEvent.ALT_MASK));
	item.addActionListener(this);
	fileMenu.add(item);
	// open
	item = new JMenuItem("Open", 'O');
        item.setAccelerator(KeyStroke.getKeyStroke('O', ActionEvent.ALT_MASK));
	item.addActionListener(this);
	fileMenu.add(item);
	// save
	item = new JMenuItem("Save", 'S');
        item.setAccelerator(KeyStroke.getKeyStroke('S', ActionEvent.ALT_MASK));
	item.addActionListener(this);
	fileMenu.add(item);
	// quit
	fileMenu.addSeparator();
	item = new JMenuItem("Quit", 'Q');
        item.setAccelerator(KeyStroke.getKeyStroke('Q', ActionEvent.ALT_MASK));
	item.addActionListener(this);
	fileMenu.add(item);
	//////// ANNOTATE MENU
	JMenu annoMenu = new JMenu("Annotation");
        annoMenu.setMnemonic('A');
	// normal annotation
	item = new JMenuItem("Annotate current document", 'A');
        item.setAccelerator(KeyStroke.getKeyStroke('C', ActionEvent.ALT_MASK));
	item.addActionListener(this);
	annoMenu.add(item);
	// batch
	item = new JMenuItem("Batch file annotation", 'B');
        item.setAccelerator(KeyStroke.getKeyStroke('B', ActionEvent.ALT_MASK));
	item.addActionListener(this);
	annoMenu.add(item);
	//////// PREFERENCES MENU
	JMenu prefMenu = new JMenu("Preferences");
        prefMenu.setMnemonic('P');
	// select model
        ButtonGroup buttonGroup = new ButtonGroup(  );
        prefMenu.add(item = new JRadioButtonMenuItem("NLPBA (5 entities)"));
        item.addActionListener(this);
	item.setSelected(true);
        buttonGroup.add(item);
        prefMenu.add(item = new JRadioButtonMenuItem("BioCreative (protein only)"));
        item.addActionListener(this);
        buttonGroup.add(item);	
	// tokenization
        prefMenu.addSeparator();
        prefMenu.add(item = new JCheckBoxMenuItem("Auto-Tokenize"));
        item.addActionListener(this);
	item.setSelected(true);
	//////// HELP MENU
	JMenu miscMenu = new JMenu("Misc");
        miscMenu.setMnemonic('M');
	// example files
	item = new JMenuItem("Demo abstract #1", '1');
	item.addActionListener(this);
	miscMenu.add(item);
	item = new JMenuItem("Demo abstract #2", '2');
	item.addActionListener(this);
	miscMenu.add(item);
	item = new JMenuItem("Demo abstract #3", '3');
	item.addActionListener(this);
	miscMenu.add(item);
	// homepage
	miscMenu.addSeparator();
	item = new JMenuItem("About ABNER");
	item.addActionListener(this);
	miscMenu.add(item);
	// put it all together
	myMenu.add(fileMenu);
	myMenu.add(annoMenu);
	myMenu.add(prefMenu);
	myMenu.add(miscMenu);
	return myMenu;
    }


    ////////////////////////////////////////////////////////////////
    // ready to rock 'n roll
    private static void createAndShowGUI() {
        //Create and set up the window.
        JFrame frame = new JFrame("ABNER v1.5");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	frame.setLocation(75,75);
        //Create and set up the content pane.
        GUI demo = new GUI();
	frame.setJMenuBar(demo.createMenuBar());
	frame.setContentPane(demo.createContent());
        //Display the window.
        frame.pack();
        frame.setVisible(true);
    }

    ////////////////////////////////////////////////////////////////
    // main
    public static void main(String[] args) {
        // Schedule a job for the event-dispatching thread: creating
        // and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
		public void run() {
		    System.err.println("Please wait while CRF models load...");
		    taggerNLPBA = new Tagger(Tagger.NLPBA);
		    taggerBIOCR = new Tagger(Tagger.BIOCREATIVE);
		    tagger = taggerNLPBA;
		    // do the GUI bit...
		    createAndShowGUI();
		}
	    });
    }
}

//################################################################
// file filter class... to help with open/save functions...
class AbnerFilter extends javax.swing.filechooser.FileFilter {
    // data members
    private String extension;
    private String description;
    // constructor
    public AbnerFilter(String ext, String desc) {
	extension = ext;
	description = desc;
    }
    // filter method
    public boolean accept(File f) {
	return f.getName().toLowerCase().endsWith(extension) || 
	    f.isDirectory();
    }
    public String getDescription() {
	return description+" file (*"+extension+")";
    }
    public String getExtension() {
	return extension;
    }
}

//################################################################
// batch converter dialog?
class BatchConverter implements ActionListener  {
    JDialog batchDialog = null;
    JFrame parent = null;
    JTextField targetDir = new JTextField(20);
    String[] returns = new String[] {".", "sgml"};
    // constructor
    public BatchConverter(JFrame p) throws java.io.IOException{
	parent = p;
	returns[0] = new File(".").getCanonicalPath();
    }
    // get the user's desired directory and output format
    public String[] getInfo() {
	batchDialog = new JDialog(parent, "Batch Converter", true);
	// set up the directory selection menu
	JPanel dir = new JPanel();
	dir.add(new JLabel("Directory: "));
	dir.add(targetDir);
	JButton dirBrowse = new JButton("Browse...");
	dirBrowse.addActionListener(this);
	dir.add(dirBrowse);
	// set up the file format combobox
	JPanel format = new JPanel();
	format.add(new JLabel("Output Format: "));
	JComboBox frmSelect = new JComboBox(new String[]{"SGML files (*.sgml)","IOB files (*.iob)","ABNER files (*.abner)"});
        frmSelect.setEditable(false);
        frmSelect.addActionListener(this);
	format.add(frmSelect);
	// set up to "OK" button
	JButton btn = new JButton("OK");
	btn.addActionListener(this);
	// add the everything...
	batchDialog.setSize(450, 130);
	batchDialog.setLocation(150, 150);
	batchDialog.getContentPane().add(dir, BorderLayout.NORTH);
	batchDialog.getContentPane().add(format, BorderLayout.CENTER);
	batchDialog.getContentPane().add(btn, BorderLayout.SOUTH);
	batchDialog.setVisible(true);
	return returns;
    }

    // the usual...
    public void actionPerformed(ActionEvent e) {
	String cmd = e.getActionCommand();
	// choose a directory
	if (cmd.equals("Browse...")) {
	    JFileChooser fc = new JFileChooser();
	    // choose directories, not files
	    fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
	    // Start in current directory
	    fc.setCurrentDirectory(new File("."));
	    // Set filter for dirs only
	    fc.setFileFilter(new AbnerFilter(".", "Directories Only"));
	    // so open chooser
	    int option = fc.showOpenDialog(parent);
	    if (option == JFileChooser.APPROVE_OPTION) {
		File sf = fc.getSelectedFile();
		returns[0] = sf.getPath();
		targetDir.setText(returns[0]);
	    }
	    else {
		System.err.println("You cancelled!");
	    }
	}
	// choose an output file format
	if (cmd.equals("comboBoxChanged")) {
	    String choice = ((String)((JComboBox)e.getSource()).getSelectedItem());
	    if (choice.startsWith("SGML"))
		returns[1] = "sgml";
	    if (choice.startsWith("IOB"))
		returns[1] = "iob";
	    if (choice.startsWith("ABNER"))
		returns[1] = "abner";
	}
	// we're done... return the bidness
	if (cmd.equals("OK"))
	    batchDialog.setVisible(false);
    }
}

//################################################################
// "about" dialog
class AboutDialog implements ActionListener {
    JDialog dialog = null;
    // constructor
    public AboutDialog(JFrame parent) {
	dialog = new JDialog(parent, "About ABNER", true);
	JButton btn = new JButton("OK");
	btn.addActionListener(this);
	dialog.getContentPane().add(btn, BorderLayout.SOUTH);
	JPanel content = new JPanel();

	ImageIcon aboutImg = new ImageIcon(GUI.class.getResource("about.gif"));
	content.add(new JLabel(aboutImg));
	/*
	content.setFont(new Font("Sans-Serif", Font.PLAIN, 7));
	content.add(new JLabel("<html><h3>ABNER: A Biomedical Named Entity Recognizer</h3></html>\n"));
	content.add(new JLabel("<html><a href='http://www.cs.wisc.edu/~bsettles/abner/'>http://www.cs.wisc.edu/~bsettles/abner/</a></html>\n"));
	content.add(new JLabel("(c) 2004 Burr Settles\n"));
	content.add(new JLabel("University of Wisconsin-Madison\n"));
	*/
	dialog.getContentPane().add(content, BorderLayout.CENTER);
	dialog.setSize(275, 315);
	dialog.setLocation(150, 150);
	dialog.setVisible(true);
    }
    // to close it
    public void actionPerformed(ActionEvent e) {
	String cmd = e.getActionCommand();
	if (cmd.equals("OK"))
	    dialog.setVisible(false);
    }
}

