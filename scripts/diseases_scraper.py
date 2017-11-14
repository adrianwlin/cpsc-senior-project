import os
import sys
import time
import random
import re
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from os.path import dirname


def remove_tags(html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', html)
    return cleantext


def parseHTML(html):
    abstracts = []
    beg = html.find("<div class=\"hidden\"")
    while beg > 0:
        end_tag = html.find(">", beg)
        end = html.find("</div>", beg)
        abstract = remove_tags(html[end_tag + 1:end])
        abstracts.append(abstract)
        beg = html.find("<div class=\"hidden\"", end)
    return abstracts


def main():
    root_folder = dirname(dirname(os.path.abspath(__file__)))
    diseases_file = os.path.join(
        root_folder, "data/human_disease_textmining_full.tsv")
    if len(sys.argv) > 1:
        diseases_file = sys.argv[1]
    outfile = open(os.path.join(root_folder, "data/training.csv"), "w")
    browser = webdriver.Chrome()
    num_abstracts = 0
    with open(diseases_file, "r") as f:
        text = f.readlines()
        num_pairs = len(text)
        random.seed(1234)
        indices = random.sample(range(num_pairs), 10)
        for i in indices:
            line = text[i]
            line_split = line.split("\t")
            ensembl_id, gene, doid, disease, z_score, confidence, url = line_split
            print gene, disease, url

            # response = urllib.urlopen(url)
            # html = response.read()
            # begin = html.find("blackmamba_pager")
            # end = html.find("</script>", begin)
            # js_request = html[begin:end]

            browser.get(url)
            timeout = 5
            try:
                element_present = EC.presence_of_all_elements_located(
                    (By.CLASS_NAME, 'hidden'))
                WebDriverWait(browser, timeout).until(element_present)
            except TimeoutException:
                print "timeout"
            abstracts = parseHTML(browser.page_source)
            num_abstracts += len(abstracts)
            for abstract in abstracts:
                outfile.write(",".join([gene, disease, abstract]))
    print num_abstracts, "abstracts scraped"


if __name__ == "__main__":
    main()
