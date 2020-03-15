""" 
    Usage: 
        python main.py --img imgPath //for image 
        python main.py --video videoPath //for video 
"""

import tensorflow as tf 
# Darkflow
from darkflow.net.build import TFNet

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import sys
from datetime import datetime 

# Darkflow TFNet used to create our plate detection model by loading trained weights
options = {"pbLoad": "../yolo-1c.pb", "metaLoad":"../yolo-1c.meta", "gpu": 0.9} 
plateDetector = TFNet(options) # model for plate Detection

# CNN devnagari character recognition
nlpCharModel = tf.keras.models.load_model('nlpCharModel.h5')

arrayOfDevnagariChar   = ("0","1","2","3","4","5","6","7","8","9","BA","PA") # array  of Devnagari character


# Function that returns prediction of the input char image 
def nepaliCharIs(predictImage):
    # predictImage is a threshold Image
    grayImage = cv2.cvtColor(predictImage, cv2.COLOR_BGR2GRAY)
    _,threshImage = cv2.threshold(grayImage,40,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resize_image = cv2.resize(threshImage,(32,32),interpolation= cv2.INTER_CUBIC)  
    resize_image = resize_image.reshape(32,32,1)
    predictArray  =np.array([resize_image/255]) # normalizing 0 to 1 
    modelPrediction  = nlpCharModel.predict(predictArray)
#     print(modelPrediction[0])
    return arrayOfDevnagariChar[np.argmax(modelPrediction[0])]


# Show the model architecture
# new_model.summary()



# Funnction thats return the segmented image using PlateDetector model 

def get_plate(p_image):
    prediction_array = plateDetector.return_predict(p_image)

    prediction_array.sort(key=lambda x: x.get('confidence'),reverse=True)
   
    high_confidence = prediction_array[0] # highest confidence 
    bottomRight  = high_confidence["bottomright"]
    topLeft = high_confidence["topleft"]
    # print(high_confidence['confidence'])

    x0 = topLeft['x'] 
    y0 = topLeft['y']
    #(x4,y4)
    xf = bottomRight['x']
    yf = bottomRight['y']
    # print("enddd")
    return p_image[y0:yf,x0:xf]  # array slicing topLeft and bottomRight coordinate


# preforms auto canny Edge detection
def auto_canny(image, sigma=0.33):
    "A lower value of sigma  indicates a tighter threshold, whereas a larger value of sigma  gives a wider threshold"
    # compute the median of the single channel pixel intensities
    v = np.median(image)
 
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper) 
    # lower threshold   ..
    # upper  threshold .. 
 
    # return the edged image
    return edged

"""
    #performs image pre procesing
        - grayyscaling
        - addpative threshold
        - autocanny edge 
    # Contour Detection
    # Top of plaate
    # bottom of plate
    # Appy nepaliCharis to predict char class
    # Return String of License Platte


""" 
def opencvReadPlate(img):
    charListBottom = []
    charListTop = []
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    thresh_inv = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,39,1)
    edges = auto_canny(thresh_inv) 
    ctrs ,_= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #    # Contour Detection
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    img_area = img.shape[0]*img.shape[1] # license plate image Area

    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi_area = w*h # contour area
        non_max_sup = roi_area/img_area # portion of area (contour Area / imageArea)

        if((non_max_sup >= 0.02) and (non_max_sup < 0.9)):
            if ((h>0.5*w) and (1.5*w>=h)): # 0.9 plus 1.5
                char = img[y:y+h,x:x+w]

                if(y>img.shape[0]*0.45):
                    charListBottom.append(nepaliCharIs(char))     # Appy nepaliCharis to predict char class

                else:
                    charListTop.append(nepaliCharIs(char))     # Appy nepaliCharis to predict char class

                # charList.append(nepaliCharIs(char))
                cv2.rectangle(img,(x,y),( x + w, y + h ),(0,255,0),3)

    licensePlateBottom="".join(charListBottom)
    licensePlateTop="".join(charListTop)
    licensePlate = licensePlateTop + " " + licensePlateBottom 
    return licensePlate     # Return String of License Platte


# compute whole process in a single image containning vehicle
def computeFrame(frame):
    licensePlate = []
    try:
        firstCropImg = get_plate(frame) #  get plate from our frame using yolo detector
        firstCropImgCopy = firstCropImg.copy()

        licensePlate.append(opencvReadPlate(firstCropImg))

        print("Extracted Plate : " + licensePlate[0])
        return licensePlate[0]

    except Exception as e:
        print(e)
        print("Error .")


#Only Image Processing
def image_fed(imagePath):
    image = cv2.imread(imagePath)
    computeFrame(image)
    cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 600,600)
    cv2.imshow('Image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Only Video Processing
def video_fed(videoPath):
    cap = cv2.VideoCapture(videoPath)
    counter=0
    d = datetime.now()
    fileName = str(d.year)+"_"+str(d.month)+"_"+str(d.day)+".csv"

    if os.path.isfile(fileName):
        logDf = pd.read_csv(fileName)
    else:
        logDf = pd.DataFrame(columns=["Time","Plate Number"])

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
#     h, w, l = frame.shape
#         frame = imutils.rotate(frame, 270)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            if counter%6== 0:
                dnow = datetime.now()
                
                plate =  computeFrame(frame)
                time= str(dnow.hour)+"-"+str(dnow.minute)+"-"+str(dnow.second)

                aDic = {"Time":time,"Plate Number":plate}
                logDf = logDf.append(aDic,ignore_index=True)

            counter+=1

            cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Video', 600,600)
            cv2.imshow('Video',frame)

            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
        else:
            cv2.waitKey(0)
            break

    logDf.to_csv(fileName,index=False)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    """ 
        Usage: 
            python main.py --img imgPath //for image 
            python main.py --video videoPath //for video 
    """

    if len(sys.argv) != 3:
        print("Not Enough Argnument : \nUsage: \n\tpython main.py --img imgPath //for image \n\tpython main.py --video videoPath //for video")
        quit()
    else:
        if sys.argv[1] == "--img" :
            # vDir = "Dataset/Vehicle/Babin_License_Plate/8.jpg"

            imagePath = sys.argv[2]
            print(f"Processing Your Image From The Path -  {imagePath}")
            if (not os.path.isfile(imagePath)) or (not os.path.exists(imagePath)):
                print(" \tInvalid  File Path")
            else:
                image_fed(imagePath)

        elif sys.argv[1] =="--video":
            # //cap = cv2.VideoCapture('../nlpr_video/2.mp4')

            videoPath = sys.argv[2]
            print(f"Processing Your Video From The Path -  {videoPath}")
            if (not os.path.isfile(videoPath)) or (not os.path.exists(videoPath)):
                print(" \tInvalid  File Path")
            else:
                video_fed(videoPath)

        else:
            print(f"Argument '{sys.argv[1]}' is not Known")
            print("Usage: \n\tpython main.py --img imgPath //for image \n\tpython main.py --video videoPath //for image")

