# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from utils.mp_utils import draw_landmarks_on_image, draw_axis


class hand_tracker:
     
     def __init__(self):
          #generic camera matrix
          self.frame_height, self.frame_width, self.channels = (480, 640, 3)
          self.focal_length = self.frame_width
          self.center = (self.frame_width/2, self.frame_height/2)
          self.camera_matrix = np.array(
                                   [[self.focal_length, 0, self.center[0]],
                                   [0, self.focal_length, self.center[1]],
                                   [0, 0, 1]], dtype = "double"
                                   )
          self.distortion = np.zeros((4, 1))
              
          # define a video capture object 
          self.vid = cv2.VideoCapture(0) 

          # STEP 2: Create an HandLandmarker object.
          base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
          options = vision.HandLandmarkerOptions(base_options=base_options,
                                                 num_hands=2)
          self.detector = vision.HandLandmarker.create_from_options(options)

          #setup initialization
          
          model_points_orig=np.zeros((21,3,100))
          for i in range(0,100):
               self.update()
               model_points_orig[:,:,i] = self.model_points
          self.model_points_avg=np.mean(model_points_orig, axis=2)
          print('INITIALIZATION COMPLETE')

     def update(self): 
          # get video frame
          ret, frame= self.vid.read() 
          self.rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          self.image = mp.Image(image_format=mp.ImageFormat.SRGB, data=self.rgb_img)
          
          # STEP 4: Detect hand landmarks from the input image.
          self.detection_result = self.detector.detect(self.image)
          
          if len(self.detection_result.handedness)>0:
               print('here')
               self.model_points = np.float32([[-l.x, -l.y, -l.z] for l in self.detection_result.hand_world_landmarks[0]])
               self.image_points = np.float32([[l.x * self.frame_width, l.y * self.frame_height] for l in self.detection_result.hand_landmarks[0]])

     def solvepose(self):
          success, self.rotation_vector, self.translation_vector = cv2.solvePnP(self.model_points_avg, self.image_points, self.camera_matrix, self.distortion, flags=cv2.SOLVEPNP_SQPNP)

     
     def visualize(self):                    
          # STEP 5: Process the classification result. In this case, visualize it.
          if len(self.detection_result.handedness)>0:
               axis_image = draw_axis(self.rgb_img, self.rotation_vector, self.translation_vector, self.camera_matrix) 
               cv2.imshow("test",cv2.cvtColor(axis_image, cv2.COLOR_RGB2BGR))
               
          annotated_image = draw_landmarks_on_image(self.image.numpy_view(), self.detection_result)   
          cv2.imshow('mp', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
          

ht = hand_tracker()

while True:
     ht.update()
     ht.solvepose()
     ht.visualize()
     if cv2.waitKey(1) & 0xFF == ord('q'): 
               break
# After the loop release the cap object 
ht.vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 

