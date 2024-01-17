# STEP 1: Import the necessary modules.
#import mediapipe as mp
#from mediapipe.tasks import python
#from mediapipe.tasks.python import vision
import cv2
#import numpy as np
from utils.mp_utils import hand_tracker

ht = hand_tracker(bimanual=True)

while True:
     ht.update()
     ht.solvepose()
     ht.visualize()
     #print(ht.rotation_matrix)
     if cv2.waitKey(1) & 0xFF == ord('q'): 
               break
# After the loop release the cap object 
ht.vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 

