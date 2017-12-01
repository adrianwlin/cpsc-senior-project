from flask import Flask, redirect, render_template, request, url_for
from werkzeug import secure_filename
import subprocess
import urllib2
from bs4 import BeautifulSoup

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']

      s = 'Found entities:\n\n'

      if f == None or not f:
        furl = request.form['file-url']
        if furl == '' or furl == None or not furl:
          return 'No File Found!'

        page = urllib2.urlopen('http://yahoo.com').read()
        soup = BeautifulSoup(page)
        soup.prettify()
        for pars in soup.findAll('p'):
          for par in pars:
            s += par.get_text()
            return s

      # tempname = 'temp/' + secure_filename(f.filename)
      # f.save(tempname)
      # subprocess.call(["python", "scripts/becasgenesdiseases.py", tempname])

      s = ""
      for line in f:
      	s += line
      	s += '\n'
      return s

if __name__ == "__main__":
    app.run()