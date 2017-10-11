import sys
import xml.etree.ElementTree as ET

def findUnique(inFile, outFile):
	seen = set() # Lines that have been seen

	# Write all unique lines to the outFile
	for line in inFile:
		if line not in seen:
			seen.add(line)
			outFile.write(line)

def main():
	# Check correct number of arguments
	if len(sys.argv) < 2:
		print("Format: python findUnique.py <inFilename>.xml")
		return 1

	# Get input file name and check validity
	inFileName = sys.argv[1]

	# Get file name without the extension
	inFileNameNoExt = inFileName
	while len(inFileNameNoExt) > 0 and inFileNameNoExt[-1] != '.':
		inFileNameNoExt = inFileNameNoExt[:-1]
	if inFileNameNoExt[-1] == '.':
		inFileNameNoExt = inFileNameNoExt[:-1]

	# Try opening the file
	try:
		inFile = open(inFileName,"r")
	except IOError:
		print("Invalid file name.")
		print("Format: python findUnique.py <inFilename>.txt")
		return 1

	# Try opening an output .txt file
	try:
		outFile = open(inFileNameNoExt + "Unique.txt","w")
	except IOError:
		print("Error opening output file.")
		return 1

	# For each line, call function to find unique elements
	findUnique(inFile, outFile)

	# Close files
	inFile.close()
	outFile.close()

	print("Found Unique.")
	return 0

if __name__ == "__main__":
	main()