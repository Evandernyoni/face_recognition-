# USAGE
# python frames.py --input videos/grei_1MP_1fps.mp4 --output video_frames/data --sec 1

# import necessary packages 
import cv2
import argparse
import os 

# construct argument parser and pass the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True)
ap.add_argument('-o', '--output', required=True)
ap.add_argument('-s', '--sec', required=True)
args = vars(ap.parse_args())


def ExtractFrames(pathInput=args['input'], pathOutput=args['output'], frameRate=args['sec']):
    '''Extracts framse from specified video

    Attributes
    ----------
    pathInput: str
             Path to video from which frames are exracted 
    pathOutput: str
               Path/name of image destination directory
    frameRate: int
             Time step at which to extract frames

    Returns
    -------
    success: bol
           boolean condition for frame existance
    frameRate: int
             Time step at which to extract frames
    
    '''
    try:
        # creating output folder
        if not os.path.exists(pathOutput):
            os.makedirs(pathOutput)
    
    # if dir is not created then raise error
    except OSError:
        print('Error: Creating the directory')
    
    # Read video from input path 
    cap = cv2.VideoCapture(pathInput)
    cap.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    
    #Capture frame-by-frame
    success , frame = cap.read()

    if success:
        print('Read %d frame: ' % count, success)
        # save frame as JPEG file
        cv2.imwrite(os.path.join(pathOutput, 'frame{:d}.png'.format(count)), frame)
    return success, frameRate


if __name__ == "__main__":
    # to extract frames, we set the parameter and counts for iteration 
    sec = 0
    count = 1 
    success, fr = ExtractFrames(pathInput=args['input'], pathOutput=args['output'], frameRate=float(args['sec']))
    # loop over the frames at specified timestep 
    while success:
        count += 1
        sec += fr
        sec = round(sec, 2)
        success, fr = ExtractFrames(pathInput=args['input'], pathOutput=args['output'], frameRate=float(args['sec']))
        





