# Overview 

This app reads an jpeg image ([...].jpg) and detect dog and human in it. 
When a dog is detected, its breed is predicted based on our deep learning model.
In case a human is detected, this app returns which dog breed she/he looks like as well as 
an image with dog ears and nose added. 

This app uses the Flask (http://flask.pocoo.org/). 
In this app, the locations to add the dog ears and nose are determined based on the locations of 
face, eyes and nose detected by OpenCV (https://pypi.python.org/pypi/opencv-python). 
For the detection, we have used the pretrained face/eye/nose detector
`haarcascade_frontalface_default.xml`, `haarcascades/haarcascade_eye.xml` 
(for face and eye, available at https://github.com/opencv/opencv/tree/master/data/haarcascades)
as well as `haarcascade_mcs_nose.xml`(for nose, available at https://github.com/opencv/opencv_contrib/blob/master/modules/face/data/cascades/haarcascade_mcs_nose.xml)


# How to use ?

1. clone this repository.

2. install the libraries summarized in `requirements.txt`

3. run `app.py` (i.e. type `python app.py` ) in the terminal. 

4. access the local address given in the terminal by using a web browser.

4. click the button and upload an image file. 
(you can use the images stored in `./static/images/` for the demonstration.)

5. when a human is detected in the image, the image with dog ears and nose added is generated. This image is stored in `./static/images/`

# Note (for future improvement)

- The detection of eyes with OpenCV is not very good (often, more than two eyes seem to be detected 
for one detected face!). This causes the error in this app. This part needs to be improved, say by introducing
better human eye detectors.

- As seen in the jupyter notebook for the project, our architecture of the deep learning model for the breed classification is simple but this classification process in the app is still slow. We must improve this point by using other algorithm.

- To upload this app to some server is another future work. (I need to learn more about AWS or Google Cloud Platform.)

