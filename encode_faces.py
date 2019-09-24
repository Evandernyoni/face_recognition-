
""" Evander Nyoni, March 28, 2019.

This programs produces 124-d vector encodings and saves them onto pickle file. 


# Usage: python encode_faces.py --dataset dataset --encodings encodings.pickle
"""

# import the necessary packages
import utilities
import argparse


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,help="path to input directory of faces + images")
ap.add_argument("-e", "--encodings", required=True,help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="cnn",help="face detection model to use: either `hog` or `cnn`")
args = vars(ap.parse_args())

# produces 124-d vector of facial encodings and saves them onto pickle file
utilities.encode_faces(inputPath=args["dataset"], method=args["detection_method"],encodingsPath=args["encodings"])
