from flask import Flask, redirect, render_template, request, url_for
from werkzeug import secure_filename
import subprocess
import urllib2
from bs4 import BeautifulSoup
from scripts.becasgenesdiseases import becasNER, printCounts
from scripts.preprocess import Preprocessor
from StringIO import StringIO
from keras.models import load_model
import sys
import io

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    # Reroute stdout to a string to print to screen
    # old_stdout = sys.stdout
    # sys.stdout = mystdout = StringIO()

    if request.method == 'POST':
        try:
            f = request.files['file']
        except:
            f = None

        print 'Found entities:\n\n'

        tempname = 'temp/tempfile.txt'

        # Check if using url or file
        # If both, use file by default
        if f == None or not f:
            # Get file url and check it
            furl = request.form['file-url']
            print furl
            if furl == '' or furl == None or not furl:
                return 'No File Found!'

            # Load up the file url
            page = urllib2.urlopen(furl).read()
            soup = BeautifulSoup(page, "html5lib")
            soup.prettify()

            # Open a file and write all the paragraph elements to it
            f2 = open(tempname, 'w')
            print "NOW GETS HERE"
            for par in soup.find_all('p'):
                f2.write(par.get_text())
            print "THEN GETS HERE"
            f2.close()
        else:
            # Using a file not a file url
            # Write the file contents to a temp
            tempname = 'temp/' + secure_filename(f.filename)
            f.save(tempname)

        # Named-entity recognition
        entityList = becasNER(tempname)

        # Sanity check
        printCounts(entityList)

        # Preprocess the data for relationship-extraction
        # Gets the features to be used by the CNN
        preprocessor = Preprocessor()
        wordEmbed, geneDist, diseaseDist, depFeatures, raw_text = preprocessor.createFeatures(
            entityList)
        print "wordEmbed", wordEmbed.shape
        print "geneDist", geneDist.shape
        print "diseaseDist", diseaseDist.shape

        if wordEmbed.shape[0]:
            # raw_text is a list of (sentence, gene, disease) tuples

            dep_model = load_model("models/model_dep.h5")
            no_dep_model = load_model("models/model_no_dep.h5")

            probsboth = no_dep_model.predict(
                [geneDist, diseaseDist, wordEmbed]).tolist()
            # probs is a numpy array of of shape (n, 2) where n is the length of raw_text

            # Put back stdout to be safe for next run
            # sys.stdout = old_stdout

            # Just the probability that the two are related
            probs = [i[1] for i in probsboth]

            # Reorder raw_text and probs by sort of probs
            inds = sorted(range(len(probs)), key=lambda k: probs[k])[::-1]

            raw_sorted = [raw_text[i] for i in inds]
            probs_sorted = [probs[i] for i in inds]

            result = []

            # Build up the result using text and probabilities
            for i in range(min(len(raw_sorted), len(probs_sorted))):
                curr = {}
                curr['data'] = raw_sorted[i]
                curr['prob'] = probs_sorted[i]
                result.append(curr)

            # Check output
            # print mystdout.getvalue()

            # return render_template("result.html", comm=mystdout.getvalue(), result=result)
            return render_template("result.html", result=result)
        print "No pairs found"
        return ('', 204)


if __name__ == "__main__":
    app.run()
