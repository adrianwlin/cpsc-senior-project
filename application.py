from flask import Flask, redirect, render_template, request, url_for
from werkzeug import secure_filename
import subprocess
import urllib2
from bs4 import BeautifulSoup
from scripts.becasgenesdiseases import becasNER, printCounts
from scripts.preprocess import createMatrices

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']

      s = 'Found entities:\n\n'

      print "FIRST REACHED THIS"

      if f == None or not f:
        furl = request.form['file-url']
        if furl == '' or furl == None or not furl:
          return 'No File Found!'

        page = urllib2.urlopen(furl).read()
        soup = BeautifulSoup(page)
        soup.prettify()
        for pars in soup.findAll('p'):
          for par in pars:
            s += par.get_text()
            return s

      print "THEN REACHED THIS"

      tempname = 'temp/' + secure_filename(f.filename)
      f.save(tempname)
      # subprocess.call(["python", "scripts/becasgenesdiseases.py", tempname])
      entityList = becasNER(tempname)
      printCounts(entityList)
      # createMatrices(entityList) # This has an error. WIll put back after

      print "NEXT REACHED THIS"

      s = ""
      for line in f:
      	s += line
      	s += '\n'
      return s

if __name__ == "__main__":
    app.run()