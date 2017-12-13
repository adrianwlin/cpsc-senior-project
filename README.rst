Biomedical Entity-Relationship Extraction - A CPSC 490 Project
=====================================================

Overview
--------

This repo is for Biomedical Entity-Relationship Extraction, a submission to and project for CPSC 490: Special Projects course at Yale University. This project was completed by Robert Tung and Adrian Lin, and was advised by Professor Dragomir Radev in the LILY Lab. The course is managed by Professor James Aspnes.

This project seeks to use natural language processing methodologies to create a pipeline that ultimately determines the probability that each pair of genes and diseases in a paper is related. This project also includes a web application that can take in a new biomedical paper, use the model trained by the pipeline, and in turn determine the probability that each pair of genes and diseases in a sentence in that paper is related.

Running the Web Application
------------

::

    export FLASK_APP=application.py
    flask run