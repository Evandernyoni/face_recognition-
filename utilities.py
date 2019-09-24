""" Evander Nyoni, March 28, 2019
This module contains all the functions for preprocessing the data, detecting faces in the input image 
and making prediction. 

soucre: https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/ 
"""

# import packages 
import cv2
import face_recognition
import pickle
import pandas as pd 
import datetime
import os
import matplotlib.pyplot as plt
from imutils import paths



def preprocessor(imagePath, detection_method):
    """ This function loads the input image and converts it from BGR to RGB. From the input image, 
        the function then detects the (x, y)-coordinates of the bounding boxes corresponding to each face
        and then computes the facial embeddings for the respective faces.   


        Parameters
        ---------- 

        imagePath: str
             path to input image

        detection_method: str
                        face detection model to be used
                         
                        
        Return
        ------
        encodings: list
                 128-d vectors

        boxes: list
            (x, y)-coordinates of the face bounding boxes

    """
    # load the input image and convert it from BGR to RGB
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # detect the (x, y)-coordinates of the bounding boxes corresponding to each face in the input image,
    #  then compute the facial embeddings for each face
    print("[INFO] recognizing faces...")
    boxes = face_recognition.face_locations(rgb, model= detection_method)
    encodings = face_recognition.face_encodings(rgb, boxes)
    
    return encodings, boxes, image


def make_prediction(test_encodings,known_encodings, boxes):
    """This function loads the known facial embadings and matches each face in the input image to our 
       known encodings. 

    Parameters
    ----------
    test_encodings: str
                face encodings from the test image
    known_encodings: pickle
                   known faces and embeddings

    boxes: list
         (x, y)-coordinates of the bounding boxes corresponding to each face in the input image


    Return
    ------
    
    names: list
         list of names for each face detected

    all_counts: dict
              dictionary of count of total number of times each face is matched
    
    counts_per_face: list
                   list of counts of total number of times each face was matched per bounding box 

    """
    # load the known faces and embeddings
    print("[INFO] loading encodings...")
    data = pickle.loads(open(known_encodings, "rb").read())


    # initialize the list of names for each face detected
    names = []
    #names2 = []
    all_counts = {}
    counts_per_face = []
    # loop over the facial embeddings 
    for i in range(len(test_encodings)):
        # match each face in the input image to our known encodings 
        matches = face_recognition.compare_faces(data["encodings"], test_encodings[i])
        name = "Unknown"
        name_best = name
   
        # check to see if we have found a match 
        if True in matches:
            # get indexes of all matched faces and initialize a dictionary to count the total
            #   number of times each face wast matched 
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
       
            # loop over the matched indexes and maintain a count for each recognized face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            
                # determine the recognized face with the largest number of votes (note: in the event of an unlikely tie Python will select first entry in the dictionary)
                #name = max(counts, key = counts.get)
                name_best = max(counts, key = counts.get)
            
                all_counts[name] = counts.get(name, 0)
            # get counts per face
            counts_per_face.append(counts)        
            
        # update the list of names  
        #names.append(name)
        names.append(name_best)
    
    return names, all_counts, counts_per_face
    

def get_probs(imagePath, all_counts, counts_per_face):
    """ This function stores the probabilities of maches per face in a data-frame and writes 
        the data-frame to a csv file. 

    Parameter
    ---------
    imagePath: str
             path to input image
    all_counts: dict
              dictionary of votings for known names. 

    Return
    ------

    probabilities: csv 
                file of prediction probabilities 
    
    """
    # Create data frame of probabilities
    df = pd.DataFrame(counts_per_face).fillna(0)
    df['image_name'] = imagePath

    now = datetime.datetime.now()
    df['time'] = now.strftime("%d-%m-%y %H:%M")

    for i in list(all_counts):
        path, dirs, files = next(os.walk("dataset/"+i))
        # To get probability we divide count by number of images in training set and multiply by accuracy of model
        df[i]=((df[i]/len(files))*0.9938)
    # Write probabilities to csv
    if not os.path.exists("probabilities.csv"):
        df.to_csv("probabilities.csv",index=False)
    else:
        df2 = pd.read_csv("probabilities.csv")
        df2 = pd.concat([df2,df],sort=True).fillna(0)
        df2.to_csv("probabilities.csv",index=False)

def bounding_box(boxes, names, image):
    """ Plots and labels bounding boxes of the recognizes faces on the input image  
    
    Parameters
    ----------

    image: numpy.ndarray
         array representation of input image

    names: list
         list of names for each face detected

    boxes: list
         (x, y)-coordinates of the bounding boxes corresponding to each face in the input image

    Return
    ------

    image: numpy.ndarray
         array representation of output image 

    """
    # loop over recognized faces 
    for ((top, right, bottom, left), name) in zip(boxes, names):
        #draw the predicted face name on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255,0), 2)
    
    return(image)

def make_plot(image):
    """ Plots the test_image with predicted faces 

    Parameters
    ----------

    image: numpy.ndarray
         array representation of output image 
    
    """
    plt.imshow(image)
    plt.savefig("output/test.png")
    plt.show()

def encode_faces(inputPath, method,encodingsPath):
    """Learbs facial encodings of input data and writes them to a serialized db of facial encodings.
    
    Parameters
    ----------
    inputPath: str
             path to input directory of faces + images

    method: str
          face detection model to use: either `hog` or `cnn`
          

    encodingsPath: str
                path to serialized db of facial encodings

    """
    
    
    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(inputPath))

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        #image = cv2.resize(image,(256,256),interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb,model= method)

        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)
    # add each encoding + name to our set of known names and encodings
    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open(encodingsPath, "wb")
    f.write(pickle.dumps(data))
    f.close()
