from flask import Flask, redirect, render_template, request, url_for
from werkzeug import secure_filename

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']

      s = ""
      for line in f:
      	s += line
      	s += '\n'
      # f.save(secure_filename(f.filename))
      # return 'file uploaded successfully'
      return s

if __name__ == "__main__":
    app.run()