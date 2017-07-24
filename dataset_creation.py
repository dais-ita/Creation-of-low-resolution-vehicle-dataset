
'''Video to frames '''
import cv2
import sys
# print(sys.version)
import datetime
import imutils
import time
import numpy as np
from pandas.io import wb

from optical_flow_functions import *

'''Projects used in the development

http://www.pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/
https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/
'''


'''loop over all videos'''

import os
video_paths = []
for root, dirs, files in os.walk("./videos"):
    for file in files:
        if file.endswith(".mp4"):
            video_paths.append(os.path.join(root, file))


'''USER DEFINED INPUTS ***********************************************'''
# Get frames from video
videoPath = video_paths[0]
frame_paths = video_to_frames(videoPath)    

# Optical flow parameters
background = None
min_area = 800 #Size of the rectangle to be captured by OpenCV
delay = 8 #Number of frames that the algorithm waits before it starts 
overlap_threshold = 0.01 #Threshold used to decide if there is overlap between two rectangles
resize_width = 500 
frame_height = 409

frame = cv2.imread(frame_paths[0],)
size = frame.shape
height = frame.shape[0]
upper_limit = frame_height *0.1
lower_limit = frame_height *0.85
left_limit = resize_width * 0.05
right_limit = resize_width * 0.95
 
trackers = [] #list containing tacker objects. Each tracker also contains a bbox 
bboxes = [] # each element of this array is a bbox = (10, 23, 86, 10) 
existing_objects = []
new_objects = []
count_not_found = 0
# print(video_paths)
for videoPath in video_paths:
    #print('VideoPath: ', videoPath)
    frame_paths = video_to_frames(videoPath)    
     
    '''Loop over the frames of the video with step 2 forward and backwards'''
    z = ['forward', 'reverse']
    for element in z:
        crop_count = 0
        if element =='forward':
            for i in range(0,len(frame_paths),2):
                #print(videoPath)
                name = '_'.join(videoPath.split('\\')[-2:])
                #print('name****************', name)
                #for i in reversed(range(0,len(frame_paths),2)):
                #print('iteration:',i)
                ''' Read frame image and apply grayscale and blur'''    
                frame, gray = get_and_preProcess_frame(frame_paths, i)
                '''Track frames and get existing objects coordinates and trackers'''
                new_objects, existing_objects, boxes_in_frame, trackers, background, ok = track_frames(trackers, background, gray, existing_objects, frame_paths, delay, upper_limit, lower_limit,
                                                                                                            right_limit, left_limit, min_area, i, frame, overlap_threshold)
                  
                #print(existing_objects)
                ''' Update tracker boxes with new positions and crop some tracker boxes'''
                bboxes, existing_objects, trackers = update_tracker(new_objects, existing_objects, boxes_in_frame, trackers, crop_count, i, gray, frame, element, name)
                      
                ''' Delete boxes close to the edge and crop some existing objects'''
                delete_boxes_and_crop_vehicles(existing_objects, crop_count, i, frame, upper_limit, lower_limit, right_limit, left_limit, bboxes, trackers, element, name)
               
                '''Draw bounding box and show video'''
                draw_bounding_box(frame, bboxes, ok)
                   
                # Exit if ESC pressed
                k = cv2.waitKey(1) & 0xff
                if k == 27 : break
                  
                  
                crop_count +=1
      
       
        else:
            for i in reversed(range(0,len(frame_paths),2)):
                name = '_'.join(videoPath.split('/')[-2:])
                #print('iteration:',i)
                ''' Read frame image and apply grayscale and blur'''    
                frame, gray = get_and_preProcess_frame(frame_paths, i)
                '''Track frames and get existing objects coordinates and trackers'''
                new_objects, existing_objects, boxes_in_frame, trackers, background, ok = track_frames(trackers, background, gray, existing_objects, frame_paths, delay, upper_limit, lower_limit,
                                                                                                            right_limit, left_limit, min_area, i, frame, overlap_threshold)
                  
                #print(existing_objects)
                ''' Update tracker boxes with new positions and crop some tracker boxes'''
                bboxes, existing_objects, trackers = update_tracker(new_objects, existing_objects, boxes_in_frame, trackers, crop_count, i, gray, frame, element, name)
                      
                ''' Delete boxes close to the edge and crop some existing objects'''
                delete_boxes_and_crop_vehicles(existing_objects, crop_count, i, frame, upper_limit, lower_limit, right_limit, left_limit, bboxes, trackers, element, name)
               
                '''Draw bounding box and show video'''
                draw_bounding_box(frame, bboxes, ok)
                   
                # Exit if ESC pressed
                k = cv2.waitKey(1) & 0xff
                if k == 27 : break
                  
                  
                crop_count +=1

    
        
    