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

/*
  Very simple example of how to use ABNER's API to tag text.
 */

import abner.*;

import java.io.*;
import java.lang.*;
import java.util.*;

public class TaggingExample {

    public static void main(String[] args) {
	Tagger t = new Tagger();
	String s = "We have identified a transcriptional repressor, Nrg1, in a genetic screen designed to reveal negative factors involved in the expression of STA1, which encodes a glucoamylase.\n\nThe NRG1 gene encodes a 25-kDa C2H2 zinc finger protein which specifically binds to two regions in the upstream activation sequence of the STA1 gene, as judged by gel retardation and DNase I footprinting analyses. Disruption of the NRG1 gene causes a fivefold increase in the level of the STA1 transcript in the presence of glucose.";

	System.out.println("################################################################");
	System.out.println(s);
	System.out.println("################################################################");
	System.out.println(t.tokenize(s));
	System.out.println("################################################################");
	System.out.println(t.tagABNER(s));
	System.out.println("################################################################");
	System.out.println(t.tagIOB(s));
	System.out.println("################################################################");
	System.out.println(t.tagSGML(s));

	System.out.println("################################################################");
	String[][] ents = t.getEntities(s);
	for (int i=0; i<ents[0].length; i++) {
	    System.out.println(ents[1][i]+"\t["+ents[0][i]+"]");
	}
	System.out.println();

	System.out.println("################################################################");
	System.out.println("[PROTEIN SEGMENTS]");
	String[] prots = t.getEntities(s,"PROTEIN");
	for (int i=0; i<prots.length; i++) {
	    System.out.println(prots[i]);
	}
	System.out.println();
    }
}
