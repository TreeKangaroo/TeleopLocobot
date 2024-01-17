# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from utils.mp_utils import draw_landmarks_on_image, draw_axis

#generic camera matrix
frame_height, frame_width, channels = (480, 640, 3)
focal_length = frame_width
center = (frame_width/2, frame_height/2)
camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
distortion = np.zeros((4, 1))

"""
class hand_tracker:
     def __init__(self, detector, side):
          self.detector=detector
          self.side = side
          
     def initialize_pose(self):
          model_points_buffer = np.zeros((21,3,100))
          for i in range(0,100):
               model_points, image_points, _=self.update(rgb_image)
               
               model_points_buffer[:,:,it]=model_points
          
          model_points_avg=np.mean(model_points_buffer, axis=2)
          
          
     def update(self, rgb_image):
          image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
     
          # STEP 4: Detect hand landmarks from the input image.
          detection_result = detector.detect(image)
          
          if len(detection_result.handedness)>0:
               model_points = np.float32([[-l.x, -l.y, -l.z] for l in detection_result.hand_world_landmarks[idx]])
               image_points = np.float32([[l.x * frame_width, l.y * frame_height] for l in detection_result.hand_landmarks[idx]])
     
     return model_points, image_points, detection_result
     
     def solvepose(self, img_points):
          success, rotation_vector, translation_vector = cv2.solvePnP(self.model_points_avg, image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_SQPNP)
"""          
# define a video capture object 
vid = cv2.VideoCapture(0) 

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

#setup initialization
itr=0
itl=0

mp_buffer_left=np.zeros((21,3,100))
mp_buffer_right=np.zeros((21,3,100))

while(True): 
     # get video frame
     ret, frame= vid.read() 
     rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
     image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
     
     # STEP 4: Detect hand landmarks from the input image.
     detection_result = detector.detect(image)
     
     if len(detection_result.handedness)>0:
          for i in range(len(detection_result.handedness)):
               if detection_result.handedness[i][0].category_name=='Right':
                    model_points_r = np.float32([[-l.x, -l.y, -l.z] for l in detection_result.hand_world_landmarks[i]])
                    image_points_r = np.float32([[l.x * frame_width, l.y * frame_height] for l in detection_result.hand_landmarks[i]])
          
                    #initialize
                    if itr<100:
                         mp_buffer_right[:,:,itr] = model_points_r
                         itr+=1
                    
                    #get average pose; this is the baseline going forward
                    if itr==100:
                         mp_right_avg=np.mean(mp_buffer_right, axis=2)
                         itr+=1
                         print('INITIALIZATION COMPLETE RIGHT')
                    
                    #tracking phase
                    if itr>100:
                         success, rotation_vector_r, translation_vector_r = cv2.solvePnP(mp_right_avg, image_points_r, camera_matrix, distortion, flags=cv2.SOLVEPNP_SQPNP)


               elif detection_result.handedness[i][0].category_name=='Left':
                    model_points_l = np.float32([[-l.x, -l.y, -l.z] for l in detection_result.hand_world_landmarks[i]])
                    image_points_l = np.float32([[l.x * frame_width, l.y * frame_height] for l in detection_result.hand_landmarks[i]])
          
                    #initialize
                    if itl<100:
                         mp_buffer_left[:,:,itl] = model_points_l
                         itl+=1
                    
                    #get average pose; this is the baseline going forward
                    if itl==100:
                         mp_left_avg=np.mean(mp_buffer_left, axis=2)
                         itl+=1
                         print('INITIALIZATION COMPLETE LEFT')
                    
                    #tracking phase
                    if itl>100:
                         success, rotation_vector_l, translation_vector_l = cv2.solvePnP(mp_left_avg, image_points_l, camera_matrix, distortion, flags=cv2.SOLVEPNP_SQPNP)

     # STEP 5: Process the classification result. In this case, visualize it.
     if len(detection_result.handedness)>0 and itr+itl>200:
          axis_image1 = draw_axis(rgb_img, rotation_vector_r, translation_vector_r, camera_matrix) 
          axis_image2 = draw_axis(axis_image1, rotation_vector_l, translation_vector_l, camera_matrix) 
          cv2.imshow("test",cv2.cvtColor(axis_image2, cv2.COLOR_RGB2BGR))
          
     annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)   
     cv2.imshow('mp', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
     if cv2.waitKey(1) & 0xFF == ord('q'): 
          break


# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 

