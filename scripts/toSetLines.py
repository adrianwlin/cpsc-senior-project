import sys

perLine = 10000
'''
This code takes a long text file / corpus of words (genes of diseases)
and creates a text file of these where every "perLine" lines is scrunched into one line.
This helps the NER run faster for testing purposes for getting performance measures.
'''
def main():
	# Check correct number of arguments
	if len(sys.argv) < 2:
		print("Format: python toSetLines.py <txtfilename>.txt")
		return 1

	# Text file to run the gene classifier on
	textFileName = None
	if len(sys.argv) >= 2:
		textFileName = sys.argv[1]
		if len(textFileName) < 4 or (textFileName[-4:] != ".txt"):
			print("Invalid text file name.")
			print("Format: python toSetLines.py <txtfilename>.txt")
			return 1

	f = open(textFileName, "r")
	f2 = open(textFileName[:-4] + 'SetLines.txt', "w")

	outLine = ''
	currLine = 0

	# For each line, group each perLine lines before putting a newline
	for line in f:
		if line[-1] == '\n':
			line = line[:-1]
		outLine += line + ' '
		currLine += 1
		if currLine >= perLine:
			outLine += '\n'
			currLine = 0

	# Write the data back out
	f2.write(outLine)

	f.close()
	f2.close()
	return 0



if __name__ == "__main__":
    main()
