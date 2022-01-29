"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""

import argparse
import json
import os
import glob
import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg", #I had to change .png extension into .jpg for this to work
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments
    #show_image(test_img,9000) Testing whether read works properly
    
    charname, charimg = enrollment(characters)
    xval, yval, widthlist, heightlist, croppedimgs, uniquelabels, labelimg = detection(test_img)
    finalimgs = []
    for i in range(1,len(uniquelabels)):
        croppedimg = croppedimgs[i-1]
        retval, threshimg = cv2.threshold(croppedimg,1,255,cv2.THRESH_BINARY_INV)
        tempfinalimg = cv2.resize(threshimg, (35,35)) #Scaling the image to a constant value
        finalimgs.append(tempfinalimg)
    
    r = recognition(charname,charimg,xval,yval,widthlist,heightlist,uniquelabels,finalimgs,characters)
    return r
    #raise NotImplementedError

def enrollment(characters):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.
    """
    --- Had to scrap the SIFT method because it kept crashing my kernel as it apparently requires more RAM ---
    sift = cv2.SIFT()
    siftkeypoint = []
    for cname, cimage in characters:
        show_image(cimage)
        kp = sift.detect(cimage,None)
        img=cv2.drawKeypoints(cimage,kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        siftkeypoint.append([cname,read_image(img),kp])
    for tempname, tempimg, tempkp in siftkeypoint:
        show_image(tempimg)
    """

    charname=[]     #change made here to store the values in an array rather than creating an array everytime
    charimg=[]
    for cname, cimage in characters:

        #charname = []
        #charimg = []
        retval, threshimg = cv2.threshold(cimage,1,255,cv2.THRESH_BINARY_INV)
        rowval, colval = np.where(threshimg == 255)
        x1 = min(rowval)
        y1 = min(colval)
        x2 = max(rowval)
        y2 = max(colval)
        croppedimg = cimage[x1:x2,y1:y2]
        scaledimg = cv2.resize(croppedimg, (35,35)) #Scaling the image to a constant value
        charname.append(cname)
        charimg.append(scaledimg)
    
    return charname, charimg
    
    """ Simple thresholding operation is used. The characters are cropped in order to get the maximum efficiency
    """
    #raise NotImplementedError

def detection(test_img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.
    """
    --- The ideas/design for the CCL Algorithm along with the Union-Find Algorithm are heavily inspired from
    https://www.youtube.com/watch?v=WNMnk9AFXeY&ab_channel=CMPT487UniversityofSaskatchewan ---
    """
    def findparent(parent,n):
     if parent[n] == n:
        return n
     if parent[n]!= n:      #removed -n
         return findparent(parent,parent[n])
    
        
    parent=[0]
    retval, threshimg = cv2.threshold(test_img,200,255,cv2.THRESH_BINARY_INV)
    show_image(threshimg)
    print(threshimg)
    print(retval)
    row, col = np.shape(test_img)
    templabelimg = np.zeros([row,col])
    labelimg = templabelimg
    count=0
    for i in range(0,row):
      for j in range(0,col):
          if threshimg[i,j] == 255:
              left = int(templabelimg[i-1,j])   #change of reversing the orders for the two
              top = int(templabelimg[i,j-1])
              if left == 0 and top == 0: #New label
                  count+=1
                  label=count
                  parent = np.append(parent,[label])
              elif left == 0 and top > 0: #Label on top
                  label = top;
              elif left > 0 and top == 0: #Label on left
                  label = left;
              elif left > 0 and top > 0: #Take minimum label
                  label = min(left,top)
                  labels = [left,top]
                  for val in labels:           #Union-Find Algorithm                   
                      if val != label:
                          fval = findparent(parent, val)
                          flabel = findparent(parent, label)
                          if fval > flabel:
                              parent[fval] = flabel
                          elif fval < flabel:
                              parent[flabel] = fval
              templabelimg[i, j] = label
    show_image(templabelimg)
    print(templabelimg)
    
    for i in range(0,row):
        for j in range(0,col):  
            if threshimg[i, j] == 255:
                labelimg[i, j]=findparent(parent,int(templabelimg[i, j]))
    show_image(labelimg)
    print(labelimg)
    croppedimgs = list()
    widthlist = list()
    heightlist = list()
    xval = list()
    yval = list()
    uniquelabels, ul = np.unique(labelimg, return_counts=True)
    print(len(uniquelabels))
    print(uniquelabels)
    print(ul)
#    show_image(labelimg)
    
    for i in range(1,len(uniquelabels)):
        rowval, colval = np.where(labelimg == uniquelabels[i])
        x1 = min(rowval)
        y1 = min(colval)
        x2 = max(rowval)
        y2 = max(colval)
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        xval.append(x1)
        yval.append(y1)
        widthlist.append(width)
        heightlist.append(height)
        croppedimg = labelimg[x1:x2+1,y1:y2+1]
        croppedimgs.append(croppedimg)
    
    return  xval, yval, widthlist, heightlist, croppedimgs, uniquelabels, labelimg
    #raise NotImplementedError

def recognition(charname,charimg,xval,yval,widthlist,heightlist,uniquelabels,finalimgs,characters):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    
    results = list()
    for i in range(0,len(finalimgs)):
        flag= 0
        floor = 7      #reduced the floor/threshold value; trail and error
        index=0
        ssdvalues = list()
        bbox = [int(xval[i]),int(yval[i]),int(widthlist[i]),int(heightlist[i])]
        for j in range(0,len(charname)):
            ssd = np.sum(((finalimgs[i])-(charimg[j]))**2)
            ssd2=ssd/4000000
            print(ssd2)
            ssdvalues.append(ssd2)
            mini = min(ssdvalues)
            if mini<floor:
                floor=mini
                flag=1
                index=j
        if flag == 0:
            results.append({"bbox":bbox ,"name":"UNKNOWN"})
        else:
            results.append({"bbox":bbox ,"name":charname[index]})
            
    return results
    #raise NotImplementedError


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
