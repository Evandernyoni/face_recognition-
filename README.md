# face_recognition_system

**Facial Recognition System (Model: ResNet 29)**

**Evander Nyoni**

This program performs facial recognition using opencv and deep learning.

Procedure (How it works):

1) Find faces in image: The first step is to detect the faces in the test image using the face\_recognition module. This module is built using dlib&#39;s state of the art face recognition which is built with deep learning and archieves an accuracy of about 99% on detecting faces.

Source: [https://pythonhosted.org/face\_recognition/readme.html](https://pythonhosted.org/face_recognition/readme.html)

2) Analyzing facial features: On the detected faces, the system then analyzes the facial features which are then stored as encordings, a 128-d vector.

3) The encodings from the test image are then compared to known facial encordings using the k-NN algorithm. For all the features in the 128-d vector and test encodings if the euclidean distance is below some threshold then system returns indicates that faces match.

**Installation and usage**

There are five directories in the root directory:

dataset: this directory contains all the &#39;training&#39; images from which we get encodings. The images are organised into sub directories based on the name.

examples: contains all the images for testing the model

output: contains all output images

videos: videos to be converted to image frames

video\_frames: ouput images from the conversion of video to images

The are four python files in the root directory:

encode\_faces.py: the 128-d vectors (encodings) are created with this script.

main.py: recognizes faces in the test image based on the created encodings.

utilities.py: this module contains functions that are used in data preprocessing and classification

frames.py: extracts image framse from videos

Running the main.py outputs an image and &#39;probabilities.csv&#39;  file.

**Installation**

Install required packages by running the following command in your terminal

- pip install -r  requirements.txt

**Training**

For simplicity we call the encoding process &#39;training&#39;. To train the model, run the following command:

- python encode\_faces.py  --dataset dataset  –encodings  encodings.pickle

To extract image framses from videos run the following command:

-     python frames.py --input videos/example.mp4 --output video\_frames/data --sec 10

**Testing**

To make predictions run the following command:

-     python main.py --encodings encodings.pickle --image examples/example\_01.png
