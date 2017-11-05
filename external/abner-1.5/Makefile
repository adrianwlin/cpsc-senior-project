# This is the makefile for ABNER (A Biomedical Named Entity
# Recognizer). It was written by Burr Settles.

####
# "ABNER" requires "MALLET" and "JLex" to be compiled and installed
# somewhere. (See the "README" file for information on where to get
# these resources.) Set these variables to their location for you.

MALLET_DIR = /u/bsettles/mallet-0.3.1
JLEX_DIR = /u/bsettles/JAVA

####
# compiler and flag variables... you may need to edit these.

#JC = javac
JV = /s/bin/java
JC = /s/bin/javac
FLAGS = \
-g:lines,vars,source -J-Xmx200m -source 1.4 \
-classpath ".:$(MALLET_DIR)/lib/mallet-deps.jar:$(MALLET_DIR)/lib/mallet.jar:$(JLEX_DIR)/"


####
# the rules

abner.jar: GUI.class ABNER-manifest
	jar cfm abner.jar ABNER-manifest *.class about.gif abner/*class abner/resources/*.crf \
	  -C $(MALLET_DIR)/class edu -C $(MALLET_DIR)/class gnu
	@ echo "#### BUILD COMPLETE. ####"

GUI.class: GUI.java abner/Tagger.class abner/Trainer.class
	$(JC) $(FLAGS) GUI.java

abner/Tagger.class: abner/Tagger.java abner/Input2TokenSequence.class abner/Scanner.class
	$(JC) $(FLAGS) abner/Tagger.java

abner/Trainer.class: abner/Trainer.java abner/Input2TokenSequence.class
	$(JC) $(FLAGS) abner/Trainer.java

abner/Input2TokenSequence.class: abner/Input2TokenSequence.java
	$(JC) $(FLAGS) abner/Input2TokenSequence.java

abner/Scanner.class: abner/Scanner.java
	$(JC) $(FLAGS) abner/Scanner.java

abner/Scanner.java: abner/scanner.jlex
	$(JV) -cp $(JLEX_DIR) JLex.Main abner/scanner.jlex
	mv abner/scanner.jlex.java abner/Scanner.java


####
# generate javadoc files

javadoc: abner/Scanner.java
	javadoc -windowtitle "Javadoc for the ABNER API" -d online/javadoc/ abner/*java

####
# generate tarball for packing up
tarball:
	tar -czvf abner-1.5.tar.gz ABNER-manifest about.gif *.java LICENSE Makefile README abner/*.*


####
# remove all generated files

clean:
	rm -f abner.jar *.class *~ abner/*.class abner/*~ abner/Scanner.java
