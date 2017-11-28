from flask import Flask, redirect, render_template, request, url_for
from werkzeug import secure_filename
import subprocess

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      tempname = 'temp/' + secure_filename(f.filename)
      f.save(tempname)
      subprocess.call(["python", "scripts/becasgenesdiseases.py", tempname])

      # s = ""
      # for line in f:
      # 	s += line
      # 	s += '\n'
      return s

if __name__ == "__main__":
    app.run()