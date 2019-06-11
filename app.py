# import libraries
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import os
from werkzeug import secure_filename

import dog_transformer
import detector

app = Flask(__name__)
app.config['DEBUG'] = True

image_folder = './static/images/'

# for index
@app.route('/')
def index():
    title = "Welcome"
    message = "Upload Your JPEG Image File"
    return render_template('index.html',message = message, title = title)

# main part
@app.route('/post', methods=['GET','POST'])
def post():
    title = 'Result'
    if request.method == 'POST':
        if not request.files['file'].filename == u'':
            message = "Upload Completed!"

            # save the uploaded image
            f = request.files['file']
            img_path = os.path.join(image_folder, secure_filename(f.filename))
            f.save(img_path)
            imgbefore = img_path

            # detect dog/human in the uploaded image
            detect = detector.dog_or_dog_like_human(img_path)

            # in case a dog is detected
            if(detect[0] == "dog"):
                label = detect[0]
                comment = detect[1]
                dogbreed = detect[2]
                imgafter = imgbefore
            # in case a human is detected
            elif(detect[0] == "human"):
                label = detect[0]
                comment = detect[1]
                dogbreed = detect[2]
                # add dog ears and nose to the image
                imgafter = dog_transformer.add_dog_parts(imgbefore)
            # in case no dog nor human is detected
            else:
                label = detect[0]
                comment = detect[1]
                dogbreed = detect[2]
                imgafter = imgbefore

        # in case the upload of the image failed
        else:
            imgbefore = []
            imgafter = []
            label = "error"
            comment = "error"
            dogbreed = "error"
            message = "Upload Failed... Try Again."

        return render_template('index.html', message = message, label = label,
                                    comment = comment, dogbreed = dogbreed,
                                    imgbefore = imgbefore, imgafter = imgafter, title = title)

    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')
