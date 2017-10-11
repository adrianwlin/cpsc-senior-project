import sys
import xml.etree.ElementTree as ET

def getGene(line, out):
	# If it could not have paragraph the link that precedes a gene name, quit
	if len(line) < len('/genes/entry.php?hgnc='):
		return

	content = '' # Paragraph content to be written

	# Iterate through the file to search for the link that precedes a gene name
	for i in range(len(line) - len('/genes/entry.php?hgnc=')):
		# Paragraph tag found
		if line[i:i+len('/genes/entry.php?hgnc=')] == '/genes/entry.php?hgnc=':
			i += len('/genes/entry.php?hgnc=') # Skip past the tag

			# Iterate through the content until closing tag found
			while i < len(line) - 4 and line[i] != '\"':
				# Increment the content
				content += line[i]
				i += 1 # Increment index
			content += '\n' # Newline between genes

	# Write content to output
	out.write(content)

def main():
	inFileName = sys.argv[1]

	# Try opening the file
	try:
		inFile = open(inFileName,"r")
	except IOError:
		print("Invalid file name.")
		print("Format: python getGenesSenescence.py <inFilename>.html")
		return 1

	# Try opening an output .txt file
	try:
		outFile = open("genes.txt","w")
	except IOError:
		print("Error opening output file.")
		return 1

	# For each line, call function to extract the paragraphs
	for line in inFile:
		getGene(line, outFile)

	print("Got genes.")
	return 0

if __name__ == "__main__":
	main()