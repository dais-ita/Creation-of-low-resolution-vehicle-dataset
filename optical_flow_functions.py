'''
Created on 20 Jun 2017

@author: QUINTANAAM_S
'''
'''Useful functions'''

import cv2
import sys
# print(sys.version)
import datetime
import imutils
import time
import numpy as np
from pandas.io import wb


def rect_area(x1,x2,y1,y2):
    return np.abs(x1-x2) * np.abs(y1-y2)

def find_boxes_in_frame(background, gray, upper_limit, lower_limit, right_limit, left_limit, min_area, frame):
    '''This function returns a list where each item corresponds to the coordinates of a box where openCV detects pixel motion'''
    # compute the absolute difference between the current frame and
    # previous + delay frame
    #threshold is 25 --> If the delta is less than 25, we discard the pixel and set it to black 
    frameDelta = cv2.absdiff(background, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    im2, cnts, hierarchy= cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes_in_frame = []
    # loop over the contours
    for c in cnts:
        
        # if the contour is too small, ignore it
        #print('contours', c)
        #print('contour area', cv2.contourArea(c))
        if cv2.contourArea(c) < min_area:
            #if not it starts at the beginning in the next iteration
            continue    
        
        '''find close contour'''
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        (x, y, w, h) = cv2.boundingRect(c)
        #print('(x, y, w, h)', (x, y, w, h))
        
        #(x,y,w,h)
        edge = False
        edge = rectangle_close_to_edge(x, y, w, h, upper_limit, lower_limit, right_limit, left_limit)
        if edge == False:
            boxes_in_frame.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)# top-left corner and bottom-right corner of rectangle
        else:
            #print('CLOSE TO EDGE', x, y, w, h)
            pass
        #print('c: ', boxes_in_frame)
    
    return boxes_in_frame


def rectangle_inside_another(new_rectangle, existing_object):
    '''Check if new_rectangle is inside existing_object'''
    [leftA, topA, wa, ha] = new_rectangle
    [leftB, topB, wb, hb] = existing_object
    rightA, bottomA = leftA + wa, topA + ha
    rightB, bottomB = leftB + wb, topB + hb
    if( topA <= topB ) or ( bottomA >= bottomB ) or ( leftA <= leftB ) or ( rightA >= rightB ):
#         print('A: ', topA, bottomA, leftA, rightA)
#         print('B: ', topB, bottomB, leftB, rightB)    #
        return False
    elif ( topA == topB ) or ( bottomA == bottomB ) or ( rightA == rightB ) or ( leftA == leftB ):
        return False
    else:
        #print('A: ', topA, bottomA, leftA, rightA)
        #print('B: ', topB, bottomB, leftB, rightB)    #
        #print("rectangle_inside_another is TRUE")
        #wait = input("PRESS ENTER TO CONTINUE.")
        return True

def find_intersection(box, existing_box, overlap_threshold, box_found):
    '''This function finds out if there is an intersection between two objects'''
    [XA1, YA1, wa, ha] = existing_box
    [XB1, YB1, wb, hb] = box
    XA2 = XA1 + wa
    XB2 = XB1 + wb
    YA2 = YA1 + ha
    YB2 = YB1 + hb
    SI= np.max([0, np.min([XA2, XB2]) - np.max([XA1, XB1])]) * np.max([0, np.min([YA2, YB2]) - np.max([YA1, YB1])])
    if SI>0:
        #print('SI', SI)
        SA = rect_area(XA1, XA2, YA1, YA2)
        SB = rect_area(XB1, XB2, YB1, YB2)
        SU = SA + SB - SI
        overlap = SI / SU
        #print ('overlap ratio: ', overlap)    
        if overlap > overlap_threshold:
            box_found = True
            return True
        else:
            return box_found
        
def find_new_objects(existing_objects, boxes_in_frame, frame, overlap_threshold):
    ''' This function checks for new objects in the frame by looking at the intersection between the new objects detected and the
        already identified as objects.
        Additionally the coordinates the existing objects are updated ??    '''
    
    new_objects = []
    #print('existing_objects', existing_objects)
    #print('boxes_in_frame', boxes_in_frame)
    boxes_exists = True
    for box in boxes_in_frame:
        box_found = False
        if existing_objects == []:
            boxes_exists = False
        if boxes_exists == True:
            i = 0
            for existing_box in existing_objects:
                inside_rect = rectangle_inside_another(box, existing_box)
                if inside_rect ==True:
                    #print(existing_objects[i], existing_objects)
                    existing_objects[i] = box
                    #print('RECTANGULO DENTRO DEL OTRO BOX IN EXISTING BOX!!!!!')
                    #print(existing_objects[i], existing_objects)
                else:
                    box_found = find_intersection(box, existing_box, overlap_threshold, box_found)
                
                i+=1
            
            if box_found == False:
                #print('box not found:', box)
                #count_not_found +=
                cv2.putText(frame, "Status: {}".format('NOT FOUND'), (box[0], box[2]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                existing_objects.append(box)
                new_objects.append(box)
        else:
            existing_objects.append(box)
            new_objects.append(box)
        #print('existing_objects', existing_objects)
    return new_objects, existing_objects

 
def object_detection(gray, existing_objects, initial, frame_paths, delay, upper_limit, lower_limit, right_limit, left_limit, min_area, i, frame, overlap_threshold):
    '''
        This function analyse the frame checks it the object detected does not exist in previous frames.
        First frame --> use as background
        Second frame --> use to identify initial objects
        @initial defines if it is the first iteration '''
    
    if initial == True:
        background = gray
        new_objects = find_boxes_in_frame(background, gray, upper_limit, lower_limit, right_limit, left_limit, min_area, frame)
        existing_objects = new_objects
        boxes_in_frame = []
    else:
        #Get background image to compare the frame with
        background = cv2.imread(frame_paths[i-delay+1],)
        background = imutils.resize(background, width=500)
        gray1 = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.GaussianBlur(gray1, (21, 21), 0)
        background = gray1
        
        boxes_in_frame = find_boxes_in_frame(background, gray, upper_limit, lower_limit, right_limit, left_limit, min_area, frame)
        
        new_objects, existing_objects = find_new_objects(existing_objects, boxes_in_frame, frame, overlap_threshold)
            
    
    return new_objects, existing_objects, boxes_in_frame

def create_trackers(new_item, trackers, frame):
    trackers.append(cv2.Tracker_create("MIL"))
    ok = trackers[-1].init(frame, new_item)
    return trackers, ok

def rectangle_close_to_edge(x, y, w, h, upper_limit, lower_limit, right_limit, left_limit):
    '''This function identifies if a rectangle that is being tracked is too close to the edge of the frame'''
    x1 = x + w
    y1 = y + h
    if (y < upper_limit) or  (y1>lower_limit) or  (x<left_limit) or  (x1>right_limit):
        #print("rectangle_close_to_edge")
        #wait = input("PRESS ENTER TO CONTINUE.")
        return True        
    else:
        return False

def crop_image(img, x,y,w,h, count):
    #Increment the size of the crop
    threshold = 10
    x, y, w, h = x - threshold, y - threshold, w + threshold, h + threshold
    
    crop_img = img[y: y + h, x: x + w] # Crop from x, y, w, h -> 100, 200, 300, 400
    # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
    #cv2.imshow("cropped", crop_img)
    #cv2.waitKey(0)
    #print(count)
#     input('crop captured')
    cv2.imwrite("./crops/%s.png" % count, crop_img)  
    #print("./crops/%s.png" % count)
    
def track_frames(trackers, background, gray, existing_objects, frame_paths, delay, upper_limit, lower_limit, right_limit, left_limit, min_area, i, frame, overlap_threshold):
    ''' Track the frames '''
    # if the first frame is None, initialize it
    new_objects = []
    boxes_in_frame = []
    ok = True
    if background is None:
        initial = True
        new_objects, existing_objects, boxes_in_frame = object_detection(gray, existing_objects, initial, frame_paths, delay, upper_limit, lower_limit, right_limit, left_limit, min_area, i, frame, overlap_threshold)
        initial = False 
        for objeto in existing_objects:
            #Create a bbox element with those coordinates
            trackers, ok = create_trackers(objeto, trackers, gray)
        background = 'not None'
    elif i>delay: #@delay is included so that we don't start checking for pixel modification after a certain number of frames
        #Start looking for new objects due to there is enough time difference to identify some pixel movement
        initial = False 
        new_objects, existing_objects, boxes_in_frame = object_detection(gray, existing_objects, initial, frame_paths, delay, upper_limit, lower_limit, right_limit, left_limit, min_area, i, frame, overlap_threshold)
        # returns [(231, 162, 311, 250), (150, 140, 243, 228), (304, 87, 343, 141)]
        for objeto in new_objects:
            #Create a bbox element with those coordinates
            #print('trackers:', trackers)
            #print('objeto:', objeto)
            trackers, ok = create_trackers(objeto, trackers, gray)
    
    return new_objects, existing_objects, boxes_in_frame, trackers, background, ok

def video_to_frames(videoPath):
    '''Video to frames'''
    vidcap = cv2.VideoCapture(videoPath)
    success,image = vidcap.read()
    count = 0
    success = True
    frame_paths = []
    while success:
        success,image = vidcap.read()
        #print('Read a new frame: ', success)
        if success == True:
            frame_paths.append("./frames/frame%d.png" % count)
            cv2.imwrite("./frames/frame%d.png" % count, image)     # save frame as JPEG file
            count += 1
          
    return frame_paths

def update_tracker(new_objects, existing_objects, boxes_in_frame, trackers, crop_count, i, gray, frame, element, name):
    '''Update tracker'''
    j=0
    bboxes = []
#     print('len(trackers)', len(trackers))
#     print('len(existing_objects)', len(existing_objects))
    for tracker_item in trackers:
        ok, bb = trackers[j].update(gray)
        (x1, y1, w1, h1) = bb
        '''CROP IMAGE'''
        if crop_count % 5 == 0:
            counter_crop = name +'_'+ str(i) + '_tracker_'+element+ '_' + str(j) 
            crop_image(frame, x1,y1,w1,h1, counter_crop)
            
        #if exists a rectangle inside this one then reduce the size
        for object in boxes_in_frame:
            inside_rect = rectangle_inside_another(object, bb)
            if inside_rect ==True:
                #replace tracker with new window size
                #object = tuple(0.9*x for x in object) 
                trackers[j] = cv2.Tracker_create("MIL")
                ok = trackers[j].init(gray, bb)
                ok, bb = trackers[j].update(gray)
                  
                bb = object
                #print('object:', object)
                #ok = trackers[j].init(gray, bb)
                  
        existing_objects[j] = bb
          
        #TODO UPDATE Existing object positions with this new position
        bboxes.append(bb)
        j+=1
            
    return bboxes, existing_objects, trackers

def delete_boxes_and_crop_vehicles(existing_objects, crop_count, i, frame, upper_limit, lower_limit, right_limit, left_limit, bboxes, trackers, element, name):
    k=0
    for existing_box in existing_objects:
        edge = False
        (x, y, w, h) = existing_box
            
        if crop_count % 5 == 0:
            '''CROP IMAGE'''
            counter_crop = name +'_'+ str(i) + '_existingBox_'+ element+ '_'+ str(k) 
            crop_image(frame, x,y,w,h, counter_crop)
        edge = rectangle_close_to_edge(x, y, w, h, upper_limit, lower_limit, right_limit, left_limit)
        if edge:
            del existing_objects[k]
            del trackers[k]
            del bboxes[k]
        k+=1

def get_and_preProcess_frame(frame_paths, i):
    '''Identify new frames'''
    #GET FRAME
    frame = cv2.imread(frame_paths[i],)
    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    return frame, gray


def draw_bounding_box(frame, bboxes, ok):
        ''' Draw bounding box '''
        for bb in bboxes:
            if ok:
                p1 = (int(bb[0]), int(bb[1]))
                p2 = (int(bb[0] + bb[2]), int(bb[1] + bb[3]))
                cv2.rectangle(frame, p1, p2, (0,0,255))
        
        # Display result
        cv2.imshow("Tracking", frame)


'''FUNCTION NOT BEING USED'''
def blob_detection():
    #     # Setup SimpleBlobDetector parameters.
#     params = cv2.SimpleBlobDetector_Params()
#      
#     # Change thresholds
# #     params.minThreshold = 10;
# #     params.maxThreshold = 200;
#      
#     # Filter by Area.
#     params.filterByArea = True
#     params.minArea = 10
#      
#     # Filter by Circularity
#     params.filterByCircularity = True
#     params.minCircularity = 0.01
#     params.maxCircularity = 0.4
#     # Filter by Convexity
#     params.filterByConvexity = False
#     params.minConvexity = 0.87
#      
#     # Filter by Inertia
#     params.filterByInertia = True
#     params.minInertiaRatio = 0
#     params.maxInertiaRatio = 0.9
#      
# #     # Create a detector with the parameters
# #     ver = (cv2.__version__).split('.')
# 
#     detector = cv2.SimpleBlobDetector_create(params)
#      
#     # Detect blobs.
#     keypoints = detector.detect(frame)
#      
#     # Draw detected blobs as red circles.
#     # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
#     im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#      
#     # Show keypoints
#     cv2.imshow("Keypoints", im_with_keypoints)
#     cv2.waitKey(0)
    pass

def remove_overlapping_boxes():
#         #remove items from list if the overlap too much

#         for existing_object in temp2_existing_objects:
#             #checks if box is inside existing_object
#             intersect_flag = rectangle_inside_another(box, existing_object)
#             if intersect_flag:
#                 del existing_objects[k]
#                 del trackers[k]
#                 del bboxes[k]
#         k+=1
    pass