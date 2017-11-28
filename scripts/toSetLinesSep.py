import sys
import random

def main():
	# Check correct number of arguments
	if len(sys.argv) < 2:
		print("Format: python toSetLinesSep.py <txtfilename>.txt")
		return 1

	# Text file to run the gene classifier on
	textFileName = None
	if len(sys.argv) >= 2:
		textFileName = sys.argv[1]
		if len(textFileName) < 4 or (textFileName[-4:] != ".txt"):
			print("Invalid text file name.")
			print("Format: python toSetLinesSep.py <txtfilename>.txt")
			return 1

	f = open(textFileName, "r")
	f2 = open(textFileName[:-4] + 'SetLinesSep.txt', "w")

	outLine = ''

	perLine = 10
	currLine = 0
	totalCount = 0

	for line in f:
		if random.randint(0, 6) != 0:
			continue
		if line[-1] == '\n':
			line = line[:-1]
		outLine += line + ' is an entity. '
		currLine += 1
		totalCount += 1
		if currLine >= perLine:
			outLine += '\n'
			currLine = 0

	f2.write(outLine)

	print "Total number of entities to detect: " + str(totalCount)

	f.close()
	f2.close()
	return 0



if __name__ == "__main__":
    main()
