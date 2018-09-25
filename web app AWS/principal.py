import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template, abort
from flask import send_file
from werkzeug import secure_filename

import predictor # all the functions to build the predictions

UPLOAD_FOLDER = "/home/ubuntu/flights"
MAIN_FOLDER = "/flights/"
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """check that the file is a csv"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/home/ubuntu/flights/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST': # when submitting upload
        # check if there is a file to be uploaded. If not, return home page.
        try:
            file = request.files['file']
        except:
            return render_template('home_page.html')

        # check if there is a file to be uploaded
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename) # upload the file (Flask way)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("found file and checked extension")
            try:
                final_output = predictor.flight_predictor(filename)
                if final_output:
                    return send_from_directory(app.config['UPLOAD_FOLDER'], final_output, as_attachment=True)
                else:
                    return render_template('error_page.html')
            except:
                return render_template('error_page.html')
        else:
            return render_template('error_page.html')
    return render_template('home_page.html')

#for production
if __name__ == '__main__':
    app.debug = True
    app.run(host = '0.0.0.0', port = 80)
