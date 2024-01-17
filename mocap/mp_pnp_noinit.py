# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
from utils.mp_utils import draw_landmarks_on_image

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
    #this was the line that was greying everything out
    #img = img.astype(np.float32)
    
    dist = np.zeros(4, dtype=float) if dist is None else dist
    points = scale * np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).reshape(-1, 3)
    axis_points, _ = cv2.projectPoints(points, rotation_vec, t, K, dist)

    img = cv2.line(img, tuple([int(i) for i in tuple(axis_points[3][:].ravel())]), tuple([int(i) for i in tuple(axis_points[0][:].ravel())]), (255, 0, 0), 3)
    img = cv2.line(img, tuple([int(i) for i in tuple(axis_points[3][:].ravel())]), tuple([int(i) for i in tuple(axis_points[1][:].ravel())]), (0, 255, 0), 3)
    img = cv2.line(img, tuple([int(i) for i in tuple(axis_points[3][:].ravel())]), tuple([int(i) for i in tuple(axis_points[2][:].ravel())]), (0, 0, 255), 3)
    return img
    
# define a video capture object 
vid = cv2.VideoCapture(0) 

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='models/hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

while(True): 
     # get video frame
     ret, frame= vid.read() 
     rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
     image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
     
     # STEP 4: Detect hand landmarks from the input image.
     detection_result = detector.detect(image)
     
     if len(detection_result.handedness)>0:
          model_points = np.float32([[-l.x, -l.y, -l.z] for l in detection_result.hand_world_landmarks[0]])
          image_points = np.float32([[l.x * frame_width, l.y * frame_height] for l in detection_result.hand_landmarks[0]])
          
          #SolvePnP
          success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_SQPNP)
          print('Rotation:   ', rotation_vector)
          print('Translation:    ', translation_vector)
          
          """
          transformation = np.eye(4)  # needs to 4x4 because you have to use homogeneous coordinates
          transformation[0:3, 3] = translation_vector.squeeze()
          # the transformation consists only of the translation, because the rotation is accounted for in the model coordinates. 
          #Take a look at this (https://codepen.io/mediapipe/pen/RwGWYJw to see how the model coordinates behave - the hand rotates, but doesn't translate

          # transform model coordinates into homogeneous coordinates
          model_points_hom = np.concatenate((model_points, np.ones((21, 1))), axis=1)

          # apply the transformation
          world_points = model_points_hom.dot(np.linalg.inv(transformation).T)
          print(world_points)
          """
     # STEP 5: Process the classification result. In this case, visualize it.
     if len(detection_result.handedness)>0:
          axis_image = draw_axis(rgb_img, rotation_vector, translation_vector, camera_matrix) 
          cv2.imshow("test",cv2.cvtColor(axis_image, cv2.COLOR_RGB2BGR))
          
     annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)   
     cv2.imshow('mp', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
     if cv2.waitKey(1) & 0xFF == ord('q'): 
          break


# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 

