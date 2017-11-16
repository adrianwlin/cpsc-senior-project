import os
import sys
import random
import pickle
import nltk
import re
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from os.path import dirname


def remove_tags(html):
    """Removes all html tags in a text
    """
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', html)
    return cleantext


def parseHTML(html):
    """All abstracts are in divs with class "hidden"
    """
    abstracts = []
    beg = html.find("<div class=\"hidden\"")
    while beg > 0:
        end_tag = html.find(">", beg)
        end = html.find("</div>", beg)
        abstract = remove_tags(html[end_tag + 1:end])
        abstracts.append(abstract)
        beg = html.find("<div class=\"hidden\"", end)
    return abstracts


def convertToDict(gene, disease, sent):
    data = {}
    data['line'] = sent
    data['genes'] = []
    data['diseases'] = []

    gene_dict = {}
    gene_dict['index'] = sent.find(gene)
    gene_dict['lengthInChars'] = len(gene)
    gene_dict['lengthInWords'] = len(gene.split(' '))
    gene_dict['name'] = gene
    data['genes'].append(gene_dict)

    dis = {}
    dis['index'] = sent.find(disease)
    dis['lengthInChars'] = len(disease)
    dis['lengthInWords'] = len(disease.split(' '))
    dis['name'] = disease
    data['diseases'].append(dis)

    return data


def scrape_abstracts(diseases_file):
    browser = webdriver.Chrome()
    num_abstracts = 0
    with open(diseases_file, "r") as f:
        text = f.readlines()
        num_pairs = len(text)
        random.seed(1234)
        indices = random.sample(range(num_pairs), 5)
        output = []
        for i in indices:
            line = text[i]
            line_split = line.split("\t")
            ensembl_id, gene, doid, disease, z_score, confidence, url = line_split
            print gene, disease, url
            browser.get(url)
            timeout = 5
            try:
                # wait until all the abstracts have loaded
                element_present = EC.presence_of_all_elements_located(
                    (By.CLASS_NAME, 'hidden'))
                WebDriverWait(browser, timeout).until(element_present)
            except TimeoutException:
                print "timeout"
                continue
            abstracts = parseHTML(browser.page_source)
            num_abstracts += len(abstracts)

            # New gene, disease pair
            # outfile.write("\t".join([gene, disease]) + "\n")
            for abstract in abstracts:
                sentences = nltk.sent_tokenize(abstract)
                for sent in sentences:
                    data_dict = convertToDict(gene, disease, sent)
                    output.append(data_dict)
                    # write one sentence per line.
                    # outfile.write(sent + "\n")
    print num_abstracts, "abstracts scraped"
    return output


def main():
    """diseases_scrpaer.py [input file from diseases website]
    """
    root_folder = dirname(dirname(os.path.abspath(__file__)))
    diseases_file = os.path.join(
        root_folder, "data/human_disease_textmining_full.tsv")
    if len(sys.argv) > 1:
        diseases_file = sys.argv[1]
    data = scrape_abstracts(diseases_file)
    out_path = os.path.join(
        root_folder, "data/diseases_training.p")
    pickle.dump(data, open(out_path, "wb"))


if __name__ == "__main__":
    main()
