import sys
import xml.etree.ElementTree as ET

def main():
	if len(sys.argv) < 2:
		print("Format: python xmltotxtparagraph.py <xmlfilename>.xml")
		return 1

	xmlFileName = sys.argv[1]
	if len(xmlFileName) < 4 or (xmlFileName[-4:] != ".xml" and xmlFileName[-5:] != ".nxml"):
		print("Invalid file name.")
		print("Format: python xmltotxtparagraph.py <xmlfilename>.xml")
		return 1

	# try:
	# 	xmlFile = open(xmlFileName,"r")
	# except IOError:
	# 	print("Invalid file name.")
	# 	print("Format: python xmltotxtparagraph.py <xmlfilename>.xml")
	# 	return 1

	try:
		txtFile = open(sys.argv[1] + ".txt","w")
	except IOError:
		print("Error opening output file.")
		return 1

	tree = ET.parse(xmlFileName)
	root = tree.getroot()
	for para in root.findall('p'):
		txt.File.write(para.text)

	# xmlFile.close()
	txtFile.close()

	print("Done.")
	return 0

if __name__ == "__main__":
	main()