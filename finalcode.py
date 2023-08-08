import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial import distance as dist
import winsound
import time


EYE_AR_THRESH = 0.3 #if the aspect ratio is less than this it will be regarded as a blink
EYE_AR_CONSEC_FRAMES = 7 
MOUTH_AR_THRESH = 0.5
SHOW_POINTS_FACE = False #The landmark point is set to false so that you have the screen with the user only
SHOW_CONVEX_HULL_FACE = False #The convex hull will be set to zero in order to see the user's eyes 
SHOW_INFO = False #Information regarding counters will be set to zero

#mou is the (mouth.aspect.ratio) ear is the (eye.aspect.ratio)
ear = 0
mou = 0

#The counter below will help with the total number of successive frames that will have an E.A.R less than EYE_EAR_Threshold and L.A.R greater than Lips_lar_threshold
CounterFRAMES_EYE = 0
CounterFRAMES_Lips = 0
CounterBLINK = 0
CounterLips = 0

def eye_aspect_ratio(eye): #This function represents the vertical distance of the eye
    d1 = dist.euclidean(eye[1], eye[5])
    d2 = dist.euclidean(eye[2], eye[4])
    d3 = dist.euclidean(eye[0], eye[3])
    return (d1 + d2) / (2.0 * d3) #E.A.R calculation for one side

def mouth_aspect_ratio(mouth):
    d1 = dist.euclidean(mouth[5], mouth[8])
    d2 = dist.euclidean(mouth[1], mouth[11])	
    d3 = dist.euclidean(mouth[0], mouth[6])
    return (d1 + d2) / (2.0 * d3) 




videoSteam = cv2.VideoCapture(0) #This will capture the webcam
ret, frame = videoSteam.read()
size = frame.shape #The image size 

detector = dlib.get_frontal_face_detector() #This will help get the facial landmarks needed
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #This Function will help predict the landmarks on the face
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] #It is the landmarks index point from left to right 
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] #It is the landmarks index point from right to left

model_points = np.array([(0.0, 0.0, 0.0),
                         (0.0, -330.0, -65.0),        
                         (-225.0, 170.0, -135.0),     
                         (225.0, 170.0, -135.0),      
                         (-150.0, -150.0, -125.0),    
                         (150.0, -150.0, -125.0)])

focal_length = size[1] #This will help us regarding the angle which will be captured from the webcam
center = (size[1]/2, size[0]/2)

cam_matrix = np.array([[focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]], dtype = "double") #This function will be used to help describe the mapping of our webcam from 3d to 24 point image

distance_coeffs = np.zeros((4,1))#Represents the distance coefficient that will be projected

t_end = time.time() #Will help to track the number of seconds which has passed since epoch (time start)
while(True):
    ret, frame = videoSteam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #This function represents the turn of the image to gray scale
    rects = detector(gray, 0)
    for rect in rects: #This function will over each of the frames and then apply facial landmark detection for each of them ,as well as determine facial landmarks
        shape = predictor(gray, rect) #Retctengle shapes in the frame 
        shape = face_utils.shape_to_np(shape)
        left = shape[lStart:lEnd] #The left eye frame's start position and end position
        right = shape[rStart:rEnd] #The right eye frame start position and end position
        jaw = shape[48:61]

        left_EAR = eye_aspect_ratio(left) #leftEar value/calclation
        right_EAR = eye_aspect_ratio(right) #rightEar value/calclation
        ear = (left_EAR + right_EAR) / 2.0 #The avarage calculation of the E.A.R for both sides
        mou = mouth_aspect_ratio(jaw)

        image_points = np.array([
                                (shape[30][0], shape[30][1]),
                                (shape[8][0], shape[8][1]),
                                (shape[36][0], shape[36][1]),
                                (shape[45][0], shape[45][1]),
                                (shape[48][0], shape[48][1]),
                                (shape[54][0], shape[54][1])
                                ], dtype="double") #This will be the images points that we'll find on our webcam, which will measure from the center of one's hairline to the chin tip /the left side and the right side of the face and visa versa


        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, cam_matrix, distance_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, cam_matrix, distance_coeffs) #The postion of the image

        if SHOW_POINTS_FACE:
            for p in image_points:
                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

        point1 = (int(image_points[0][0]), int(image_points[0][1]))
        point2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        
        if SHOW_CONVEX_HULL_FACE: #This function will help to visualize the facial landmarks for the eye regions
            left_Eye_Hull = cv2.convexHull(left)
            right_Eye_Hull = cv2.convexHull(right)
            jaw_Hull = cv2.convexHull(jaw)

            

            cv2.drawContours(frame, [left_Eye_Hull], 0, (255, 255, 255), 1)# This function will draw the boundary points of the eyes convex hull 
            cv2.drawContours(frame, [right_Eye_Hull], 0, (255, 255, 255), 1)
            cv2.drawContours(frame, [jaw_Hull], 0, (255, 255, 255), 1)
            cv2.line(frame, point1, point2, (255,255,255), 2)#This will help us draw image lines coordinates 


        if point2[1] > point1[1]*1.5 or CounterBLINK > 5 or CounterLips > 3: #This function will determine if a user is sleepy, if any of the conditions are true it will alert the user
            cv2.putText(frame, "The Driver is Drowsy!!!", (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) #Display the text
            
        if ear < EYE_AR_THRESH:
            CounterFRAMES_EYE += 1

            if CounterFRAMES_EYE >= EYE_AR_CONSEC_FRAMES:#This function will alert you that the driver is tired as the COUNTER_FRAMES_EYE is less than equal to EYE_EAR_CONSEC_FRAMES which means the driver frames are below what is supposed to be 
                cv2.putText(frame, "SLEEP ALERT!!", (200, 30), #Display the text
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                winsound.Beep(449, 500) #Beep to alert the driver that he/she is  falling asleep
        else:
            if CounterFRAMES_EYE > 2:
                CounterBLINK += 1
            CounterFRAMES_EYE = 0 #A normal blink
        
        if mou >= MOUTH_AR_THRESH:
            CounterFRAMES_Lips += 1
        else:
            if CounterFRAMES_Lips > 5:
                CounterLips += 1
      
            CounterFRAMES_lips = 0 #A normal yawn 
        
        if (time.time() - t_end) > 60: #Rest the blink number and count number back to 0
            t_end = time.time()
            CounterBLINK = 0
            CounterLips = 0
        
    if SHOW_INFO: #This will help draw the yawns and blink (EAR and LAR)
        cv2.putText(frame, "E.A.R: {:.2f}".format(ear), (30, 450),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "M.A.R: {:.2f}".format(mou), (200, 450),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Blinks: {}".format(CounterBLINK), (10, 30), #Display blink the number of blinks 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, "Lips: {}".format(CounterLips), (10, 60), #Lip counter number of times yawned 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Output", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key == ord('p'):
        SHOW_POINTS_FACE = not SHOW_POINTS_FACE #Dont't display landmark point of face
    if key == ord('c'):
        SHOW_CONVEX_HULL_FACE = not SHOW_CONVEX_HULL_FACE #Dont't display eye covex hull 
    if key == ord('i'):
        SHOW_INFO = not SHOW_INFO #Don't display counter information of ear,lips ,mou
    time.sleep(0.02)
    
videoSteam.release()  
cv2.destroyAllWindows()