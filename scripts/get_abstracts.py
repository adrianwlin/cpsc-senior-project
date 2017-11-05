import os
import sys


def writeAbstracts(line, out):
    beg = line.find("<abstract>")
    end = line.find("</abstract>")
    if beg < 0:
        return False
    elif end > 0:
        abstract = line[beg + len("<abstract>"):end]
        content = ''
        tagOpen = False
        for char in abstract:
            if char == "<" and not tagOpen:
                tagOpen = True
            elif char == ">" and tagOpen:
                tagOpen = False
            elif tagOpen:
                continue
            elif not tagOpen:
                content += char
        out.write(content)
        out.write("\n")
        return True
    return False


def main():
    txtFile = open("abstracts.txt", "w")
    if len(sys.argv) < 2:
        print("Format: python get_abstracts.py num_files")
        return 1
    num_files = int(sys.argv[1])
    n = 0
    # Get input file name and check validity
    for dirpath, dirs, files in os.walk("."):
        for filename in files:
            if len(filename) > 4 and filename[-5:] == ".nxml":
                try:
                    xmlFile = open(os.path.join(dirpath, filename), "r")
                except IOError:
                    print("could not open" + filename)
                    continue
                for line in xmlFile:
                    if writeAbstracts(line, txtFile):
                        n += 1
                if n >= num_files:
                    txtFile.close()
                    return 0
    txtFile.close()
    return 0


if __name__ == "__main__":
    main()
