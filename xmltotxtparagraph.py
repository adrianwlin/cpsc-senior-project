import sys
import xml.etree.ElementTree as ET

def printParagraphs(line):
	if len(line) < 7:
		return

	content = ''

	for i in range(len(line) - 7):
		if line[i:i+3] == '<p>':
			i += 3
			while i < len(line) - 3 and line[i:i+4] != '</p>':
				if line[i] == '<':
					while(line[i] != '>'):
						i += 1
					i += 1
				content += line[i]
				i += 1
			content += ' '

	print(content)

def main():
	if len(sys.argv) < 2:
		print("Format: python xmltotxtparagraph.py <xmlfilename>.xml")
		return 1

	xmlFileName = sys.argv[1]
	if len(xmlFileName) < 4 or (xmlFileName[-4:] != ".xml" and xmlFileName[-5:] != ".nxml"):
		print("Invalid file name.")
		print("Format: python xmltotxtparagraph.py <xmlfilename>.xml")
		return 1

	xmlFileNameNoExt = xmlFileName
	while len(xmlFileNameNoExt) > 0 and xmlFileNameNoExt[-1] != '.':
		xmlFileNameNoExt = xmlFileNameNoExt[:-1]
	if xmlFileNameNoExt[-1] == '.':
		xmlFileNameNoExt = xmlFileNameNoExt[:-1]

	try:
		xmlFile = open(xmlFileName,"r")
	except IOError:
		print("Invalid file name.")
		print("Format: python xmltotxtparagraph.py <xmlfilename>.xml")
		return 1

	# content = '\n'.join(xmlFile.readlines())
	# print(content)

	try:
		txtFile = open(xmlFileNameNoExt + ".txt","w")
	except IOError:
		print("Error opening output file.")
		return 1

	for line in xmlFile:
		printParagraphs(line)		

	# root = ET.fromstring(content)
	# # root = tree.getroot()
	# for para in root.findall('p'):
	# 	print("here")
	# 	print(para.txt)
	# 	# txtFile.write(para.text)

	xmlFile.close()
	txtFile.close()

	print("Done.")
	return 0

if __name__ == "__main__":
	main()