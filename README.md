# cpsc-senior-project

Gene entity sources: http://genomics.senescence.info/genes/allgenes.php, https://en.wikipedia.org/wiki/List_of_human_genes, http://ideonexus.com/2008/05/13/the-top-10-human-genes/, http://emptypipes.org/2014/12/08/gene-popularity/

How genes are named: https://ghr.nlm.nih.gov/primer/mutationsanddisorders/naming

Note that there is an existing list of Human genes, but to check this for every word in a text would be far too time-intensive (O(mn) where n is the length of the text and m is the number of human genes). Thus, we build a classification model from these genes and use that.

Got the full list of human gene names here: http://www.ensembl.org/biomart/martview/4adf27e2d42c7e363f522f1e3c3ee05a

NCBI: https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/

NER accuracy paper: https://www.ncbi.nlm.nih.gov/pubmed/25946862 (For recognitions of both gene and protein names, we achieved 97.2% for precision (P), 95.2% for recall (R), and 96.1 for F-measure. While for protein names recognition we gained 98.1% for P, 97.5% for R and 97.7 for F-measure.)

https://watermark.silverchair.com/357.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAbcwggGzBgkqhkiG9w0BBwagggGkMIIBoAIBADCCAZkGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMR7FdEZmE-Olt2iWEAgEQgIIBavXATNTXrArIvsLf37IpLvt_GU9Fcpa3CdKDJRY8OStkWrUg9vGQW9hnHX7WJIn22hlaIjrYbTwKzvzPJN4ftxe9liHtyi9spzZO1sH22emRGNPw3Y1UjJ9w3FMkBBzfmrurMFgxBNF3cNZU8dpZ4shSkIob3FpdrkAsjpu4HqxXMfW-F8qoDqZs_IVvcKwXL4h6Lf8rYQOFVFxjJ2G76GGgYZlfow82QfZipJJrZQLQdpLfJ6B3V2BvXA4eLAVPsPHo5wLZH6blb00zs8eaJg1asXyyIyZyYitgZ8Vih9fiQJTjL8yq2oKahooJqrCEGEOT6D6rLvsPgE4yMkXBDh6IOp_awtP2SH7HgaLp0NNLiOoqi7u-q4oh90kFTqNmyDE9lj0gj8ou5t47Rk9AmeH1i5CkqGvaTaUrV0FXzv-iH7fk2oLcFRPM9KGy_ASTo64a_Axf4ET49gISpOxm7W-wMZsEgYlfN0zP:
(Considering that the best current NER
systems reach an F-measure around 85 per
cent, there is a real danger that all systems
reporting better results will only represent
an overfitting of the method to the
particular gold standard, ie annotator)

becas was validated against CRAFT, AnEM and the NCBI Diseases corpora, achieving f-measure results for overlap matching of 76% for genes and proteins, and 85% for diseases. These results are on par with current state-of-the-art biomedical annotation systems.