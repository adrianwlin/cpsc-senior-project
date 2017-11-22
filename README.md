# cpsc-senior-project

Gene entity sources: http://genomics.senescence.info/genes/allgenes.php, https://en.wikipedia.org/wiki/List_of_human_genes, http://ideonexus.com/2008/05/13/the-top-10-human-genes/, http://emptypipes.org/2014/12/08/gene-popularity/

How genes are named: https://ghr.nlm.nih.gov/primer/mutationsanddisorders/naming

Note that there is an existing list of Human genes, but to check this for every word in a text would be far too time-intensive (O(mn) where n is the length of the text and m is the number of human genes). Thus, we build a classification model from these genes and use that.

Got the full list of human gene names here: http://www.ensembl.org/biomart/martview/4adf27e2d42c7e363f522f1e3c3ee05a

NCBI: https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/

NER accuracy paper: https://www.ncbi.nlm.nih.gov/pubmed/25946862 (For recognitions of both gene and protein names, we achieved 97.2% for precision (P), 95.2% for recall (R), and 96.1 for F-measure. While for protein names recognition we gained 98.1% for P, 97.5% for R and 97.7 for F-measure.)