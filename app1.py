import cv2
import numpy as np
import time

cap=cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_file =cv2.VideoWriter("output.avi",fourcc,20.0,(640,480))

time.sleep(2)
bg=0
# capturing the background
for i in range(60):
    ret,bg= cap.read()
    if not ret: 
        print("Cant receive image")
        break
    bg=np.flip(bg,axis=1)

#capturing images

while(cap.isOpened()):
    ret,img=cap.read()
    if not ret: break
    img=np.flip(img,axis=1)

    hsv= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #generating mask
    lower_red=np.array([0,120,70])
    upper_red=np.array([10,255,255])

    mask1= cv2.inRange(hsv,lower_red,upper_red)

    lower_red=np.array([170,120,70])
    upper_red=np.array([180,255,255])

    mask2= cv2.inRange(hsv,lower_red,upper_red)

    mask1=mask1+mask2

    mask1=cv2.morphologyEx(mask1, cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    mask1=cv2.morphologyEx(mask1, cv2.MORPH_DILATE,np.ones((3,3),np.uint8))

    mask2= cv2.bitwise_not(mask1)

    res1= cv2.bitwise_and(img,img,mask=mask2)
    res2= cv2.bitwise_and(bg,bg,mask=mask1)

    final_output= cv2.addWeighted(res1,1,res2,1,0)
    out_file.write(final_output)
    cv2.imshow("magic",final_output)
    cv2.waitKey(1)

cap.release()
out_file.release()
cv2.destroyAllWindows()


