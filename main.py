""" Evander Nyoni, March 28, 2019
This is the main scipt for the facial recognition system and is responsible for calling and running the following:

1. preprocessing the data
2. detecting faces in the input image 
3. making prediction and giving its probabilities  

"""

# Usage: python main.py --encodings encodings.pickle --image examples/example_02.png

# import packages 
import argparse
import utilities 


# construct the argument parser and parse it's arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-d", "--detection_method", type=str, default="cnn", help="face detection model to use: either 'hog' or 'cnn'")
ap.add_argument("-p", "--show_plot", type=str, default="y", help="show plot when done?: use either 'y' or 'n'" )
args = vars(ap.parse_args())

def classify(imagePath,encodingsPath,method,show_plot):
    """This function runs the facial regonition system by preprocessing the input image, detecting 
       faces and making preditions.

    Parameters
    ----------
    imagePath: str
            path to input image
    encodingsPath: str
                path to serialized db of facial encodings
    method: str
          face detection model to use: either 'hog' or 'cnn'

    """
    # preprocess the data
    encodings, boxes, image = utilities.preprocessor(imagePath, method)

    # predict faces 
    names, all_counts, counts_per_face = utilities.make_prediction(encodings, encodingsPath, boxes)

    # create csv file of probabilities 
    utilities.get_probs(imagePath, all_counts, counts_per_face)

    # plot bounding boxes
    image = utilities.bounding_box(boxes, names,image)

    # show plot 
    if show_plot == 'y':
        utilities.make_plot(image)
    else: 
        exit()
    

if __name__ == "__main__":
    
    classify(imagePath=args['image'],encodingsPath=args['encodings'],method=args['detection_method'], show_plot=args['show_plot'])



