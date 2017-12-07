import sys
import xml.etree.ElementTree as ET

'''
Take in a line and extract all non-tag text between paragraph tags.
Write this back out to a file.
'''
def writeParagraphsToFile(line, out):
	# If it could not have paragraph opening and closing tags, quit
	if len(line) < 7:
		return

	content = '' # Paragraph content to be written

	# Iterate through the file to search for paragraph tags
	for i in range(len(line) - 7):
		# Paragraph tag found
		if line[i:i+3] == '<p>':
			i += 3 # Skip past the tag

			# Iterate through the content until closing tag found
			while i < len(line) - 3 and line[i:i+4] != '</p>':
				# Ignore any inner tags
				if line[i] == '<':
					while(line[i] != '>'):
						i += 1
					i += 1

				# Increment the content
				content += line[i]
				i += 1 # Increment index
			content += ' ' # Space between paragraphs

	# Write content to output
	out.write(content)

def main():
	# Check correct number of arguments
	if len(sys.argv) < 2:
		print("Format: python xmltotxtparagraph.py <xmlfilename>.xml")
		return 1

	# Get input file name and check validity
	xmlFileName = sys.argv[1]
	if len(xmlFileName) < 4 or (xmlFileName[-4:] != ".xml" and xmlFileName[-5:] != ".nxml"):
		print("Invalid file name.")
		print("Format: python xmltotxtparagraph.py <xmlfilename>.xml")
		return 1

	# Get file name without the extension
	xmlFileNameNoExt = xmlFileName
	while len(xmlFileNameNoExt) > 0 and xmlFileNameNoExt[-1] != '.':
		xmlFileNameNoExt = xmlFileNameNoExt[:-1]
	if xmlFileNameNoExt[-1] == '.':
		xmlFileNameNoExt = xmlFileNameNoExt[:-1]

	# Try opening the file
	try:
		xmlFile = open(xmlFileName,"r")
	except IOError:
		print("Invalid file name.")
		print("Format: python xmltotxtparagraph.py <xmlfilename>.xml")
		return 1

	# Try opening an output .txt file
	try:
		txtFile = open(xmlFileNameNoExt + ".txt","w")
	except IOError:
		print("Error opening output file.")
		return 1

	# For each line, call function to extract the paragraphs
	for line in xmlFile:
		writeParagraphsToFile(line, txtFile)

	# Close files
	xmlFile.close()
	txtFile.close()

	print("XML Parsed.")
	return 0

if __name__ == "__main__":
	main()