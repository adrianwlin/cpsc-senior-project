from flask import Flask, redirect, render_template, request, url_for
from werkzeug import secure_filename
import subprocess
import urllib2
from bs4 import BeautifulSoup
from scripts.becasgenesdiseases import becasNER, printCounts
from scripts.preprocess import Preprocessor
from StringIO import StringIO
from keras import load_model
import sys

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    # Reroute stdout to a string to print to screen
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    if request.method == 'POST':
        f = request.files['file']

        print 'Found entities:\n\n'

        tempname = 'tempfile.txt'

        # Check if using url or file
        # If both, use file by default
        if f == None or not f:
            # Get file url and check it
            furl = request.form['file-url']
            if furl == '' or furl == None or not furl:
                return 'No File Found!'

            # Load up the file url
            page = urllib2.urlopen(furl).read()
            soup = BeautifulSoup(page)
            soup.prettify()

            # Open a file and write all the paragraph elements to it
            f2 = open(tempname, w)
            for pars in soup.findAll('p'):
                for par in pars:
                    f2.write(par.get_text())
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
        # raw_text is a list of (sentence, gene, disease) tuples

        dep_model = load_model("models/model_dep.h5")
        no_dep_model = load_model("models/model_no_dep.h5")

        probs = no_dep_model.predict([geneDist, diseaseDist, wordEmbed])
        # probs is a numpy array of of shape (n, 2) where n is the length of raw_text

        print "NEXT REACHED THIS"

        # Put back stdout to be safe for next run
        sys.stdout = old_stdout

        # Check output
        print mystdout.getvalue()

        return render_template("result.html", comm=mystdout.getvalue())


if __name__ == "__main__":
    app.run()
