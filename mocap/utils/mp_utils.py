
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
import mediapipe as mp
import numpy as np
import cv2
from mediapipe.tasks.python import vision
import time
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def draw_world_landmarks(detection_result):
  hand_landmarks_list = detection_result.hand_world_landmarks
  handedness_list = detection_result.handedness


  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.plot_landmarks(
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_utils.DrawingSpec(color=(255,0,0), thickness=5),
      solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=5),
      10, 10
    )


def extract_keypoint(detection_result, idx):
  hand_landmarks_list = detection_result.hand_world_landmarks
  if len(hand_landmarks_list)>0:
    landmark = hand_landmarks_list[0][idx]
    return [landmark.x, landmark.y, landmark.z]
  else:
    return None

def draw_axis(img, rotation_vec, t, K, scale=0.1, dist=None):
    """
    Draw a 6dof axis (XYZ -> RGB) in the given rotation and translation
    :param img - rgb numpy array
    :rotation_vec - euler rotations, numpy array of length 3,
                    use cv2.Rodrigues(R)[0] to convert from rotation matrix
    :t - 3d translation vector, in meters (dtype must be float)
    :K - intrinsic calibration matrix , 3x3
    :scale - factor to control the axis lengths
    :dist - optional distortion coefficients, numpy array of length 4. If None distortion is ignored.
    """
    #this was the line that was whiteing everything out
    #img = img.astype(np.float32)
    
    dist = np.zeros(4, dtype=float) if dist is None else dist
    points = scale * np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)
    axis_points, _ = cv2.projectPoints(points, rotation_vec, t, K, dist)

    img = cv2.line(img, tuple([int(i) for i in tuple(axis_points[3][:].ravel())]), tuple([int(i) for i in tuple(axis_points[0][:].ravel())]), (255, 0, 0), 3)
    img = cv2.line(img, tuple([int(i) for i in tuple(axis_points[3][:].ravel())]), tuple([int(i) for i in tuple(axis_points[1][:].ravel())]), (0, 255, 0), 3)
    img = cv2.line(img, tuple([int(i) for i in tuple(axis_points[3][:].ravel())]), tuple([int(i) for i in tuple(axis_points[2][:].ravel())]), (0, 0, 255), 3)
    return img
    
class hand_tracker:
     
     def __init__(self, bimanual=False):
          #generic camera matrix
          self.bimanual = bimanual
          print('INITIALIZING')
          
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
          base_options = python.BaseOptions(model_asset_path='./mocap/models/hand_landmarker.task')
          options = vision.HandLandmarkerOptions(base_options=base_options,
                                                 num_hands=2)
          self.detector = vision.HandLandmarker.create_from_options(options)

          #setup initialization
          if self.bimanual:
               self.mp_buffer_left=np.zeros((21,3,100))
               self.mp_buffer_right=np.zeros((21,3,100))
               for i in range(0,100):
                    self.update()
                    self.mp_buffer_right[:,:,i] = self.model_points_r
                    self.mp_buffer_left[:,:,i] = self.model_points_l
               self.mp_left_avg=np.mean(self.mp_buffer_left, axis=2)
               self.mp_right_avg=np.mean(self.mp_buffer_right, axis=2)
               print('INITIALIZATION COMPLETE')
               
          else:
               print('here!!!!!!!')
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
               if self.bimanual:
                    for i in range(len(self.detection_result.handedness)):
                         if self.detection_result.handedness[i][0].category_name=='Right':
                              self.model_points_r = np.float32([[-l.x, -l.y, -l.z] for l in self.detection_result.hand_world_landmarks[i]])
                              self.image_points_r = np.float32([[l.x * self.frame_width, l.y * self.frame_height] for l in self.detection_result.hand_landmarks[i]])
                         
                         elif self.detection_result.handedness[i][0].category_name=='Left':
                              self.model_points_l = np.float32([[-l.x, -l.y, -l.z] for l in self.detection_result.hand_world_landmarks[i]])
                              self.image_points_l = np.float32([[l.x * self.frame_width, l.y * self.frame_height] for l in self.detection_result.hand_landmarks[i]])
               else:
                    self.model_points = np.float32([[-l.x, -l.y, -l.z] for l in self.detection_result.hand_world_landmarks[0]])
                    self.image_points = np.float32([[l.x * self.frame_width, l.y * self.frame_height] for l in self.detection_result.hand_landmarks[0]])

     def solvepose(self):
          if self.bimanual:
               success, self.rotation_vector_r, self.translation_vector_r = cv2.solvePnP(self.mp_right_avg, self.image_points_r, self.camera_matrix, self.distortion, flags=cv2.SOLVEPNP_SQPNP)
               success, self.rotation_vector_l, self.translation_vector_l = cv2.solvePnP(self.mp_left_avg, self.image_points_l, self.camera_matrix, self.distortion, flags=cv2.SOLVEPNP_SQPNP)
               self.rotation_matrix_r, _ = cv2.Rodrigues(self.rotation_vector_r)
               self.rotation_matrix_l, _ = cv2.Rodrigues(self.rotation_vector_l)
          else:
               success, self.rotation_vector, self.translation_vector = cv2.solvePnP(self.model_points_avg, self.image_points, self.camera_matrix, self.distortion, flags=cv2.SOLVEPNP_SQPNP)
               self.rotation_matrix, _ = cv2.Rodrigues(self.rotation_vector)
     
     def visualize(self):                    
          # STEP 5: Process the classification result. In this case, visualize it.
          if len(self.detection_result.handedness)>0:
               
               if self.bimanual:
                    axis_image1 = draw_axis(self.rgb_img, self.rotation_vector_r, self.translation_vector_r, self.camera_matrix) 
                    axis_image2 = draw_axis(axis_image1, self.rotation_vector_l, self.translation_vector_l, self.camera_matrix) 
                    cv2.imshow("test",cv2.cvtColor(axis_image2, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                    
               else:
                    axis_image = draw_axis(self.rgb_img, self.rotation_vector, self.translation_vector, self.camera_matrix) 
                    cv2.imshow("test",cv2.cvtColor(axis_image, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
               
         # annotated_image = draw_landmarks_on_image(self.image.numpy_view(), self.detection_result)   
          #cv2.imshow('mp', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
