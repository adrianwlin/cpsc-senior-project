import os
import sys
import random
import pickle
import nltk
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from os.path import dirname
"""
This script scrapes the DIESASES tool for paper abstracts which mention gene
and disease pairs and write out sentences in the abstracts which contain both a
gene and a disease. The tool can be found at diseases.jensenlab.org.

diseases_scraper.py [in_file] [out_file] [start_index] [num_sentences]

The in_file is supplied by the DISEASES website
(diseases.jensenlab.org/Downloads) and is a tab separated file with each line
containing (gene ensembl id, gene, diease DOID, disease, z_score, confidence, url)
tuples.

The out_file is a txt file with one sentence per line.

Before the abstracts are scraped, the rows of the in_file are scrambled with a
a set random seed (1234). start_index specifies the starting index in the
shuffled ordering. This argument is used when you want to scrape new data that
does not overlap with the data you already have.

num_sentences specifies the number of sentences that you want.
"""


def remove_tags(html):
    """Removes all html tags in a text
    """
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', html)
    return cleantext


def extractTokens(abstract):
    """Returns two sets of gene and disease tokens based on the diesase page tags
    """
    gene_tokens = set()
    disease_tokens = set()
    soup = BeautifulSoup(abstract, "html.parser")
    genes = soup.find_all('span', {'class': 'document_match_type1'})
    for gene in genes:
        gene_tokens.add(gene.get_text())
    diseases = soup.find_all('span', {'class': 'document_match_type2'})
    for disease in diseases:
        disease_tokens.add(disease.get_text())
    return gene_tokens, disease_tokens


def parseHTML(html):
    """All abstracts are in divs with class "hidden"
    """
    abstracts = []
    beg = html.find("<div class=\"hidden\"")
    gene_tokens = set()
    disease_tokens = set()
    while beg > 0:
        end_tag = html.find(">", beg)
        end = html.find("</div>", beg)
        abstract = html[end_tag + 1:end]

        genes, diseases = extractTokens(abstract)
        gene_tokens = gene_tokens.union(genes)
        disease_tokens = disease_tokens.union(diseases)

        abstract = remove_tags(abstract)
        abstracts.append(abstract)
        beg = html.find("<div class=\"hidden\"", end)
    return abstracts, gene_tokens, disease_tokens


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


def scrape_abstracts(diseases_file, out_file, start_index, max_sentences):
    """writes to a text file with tab separated tokens
    gene, diseases, sentence mentioning the gene and disease"""
    num_abstracts = 0
    num_sent = 0
    pairs_processed = 0
    out = open(out_file, "w")
    with open(diseases_file, "r") as f:
        text = f.readlines()
        num_pairs = len(text)
        random.seed(1234)
        shuffled_indices = range(num_pairs)
        random.shuffle(shuffled_indices)
        # indices = random.sample(range(num_pairs), 3)
        output = []
        browser = webdriver.Chrome()
        for i in shuffled_indices[start_index:]:
            pairs_processed += 1
            line = text[i]
            line_split = line.split("\t")
            ensembl_id, gene, doid, disease, z_score, confidence, url = line_split
            # print gene, disease, url
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
            abstracts, gene_tokens, disease_tokens = parseHTML(
                browser.page_source)
            num_abstracts += len(abstracts)

            # New gene, disease pair
            for abstract in abstracts:
                sentences = nltk.sent_tokenize(abstract)
                for sent in sentences:
                    gene_found = False
                    disease_found = False
                    for gene in gene_tokens:
                        if sent.find(gene) >= 0:
                            gene_token = gene
                            gene_found = True
                            break
                    for dis in disease_tokens:
                        if sent.find(dis) >= 0:
                            disease_token = dis
                            disease_found = True
                            break
                    if gene_found and disease_found:
                        out.write(sent.encode('utf-8') + "\n")
                        # out.write(
                        #     "\t".join([gene_token, disease_token, sent]).encode('utf-8') + "\n")
                        num_sent += 1
                        if num_sent == max_sentences:
                            print num_pairs, "disease/gene pairs processed"
                            print num_abstracts, "abstracts scraped"
                            print num_sent, "sentences written"
                            return
            print "sentence count: ", num_sent, "pairs processed: ", pairs_processed
    print num_pairs, "disease/gene pairs processed"
    print num_abstracts, "abstracts scraped"
    print num_sent, "sentences written"


def main():
    """diseases_scraper.py [in_file] [out_file] [start_index] [num_sentences]
    """
    if len(sys.argv) != 5:
        print "diseases_scraper.py [in_file] [out_file] [start_index] [num_sentences]"
        return 1
    else:
        in_file = os.path.abspath(sys.argv[1])
        out_file = os.path.abspath(sys.argv[2])
        start_index = int(sys.argv[3])
        max_sentences = int(sys.argv[4])
    scrape_abstracts(in_file, out_file, start_index, max_sentences)


if __name__ == "__main__":
    main()
