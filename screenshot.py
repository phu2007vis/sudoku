import cv2
import pyautogui
import numpy as np
import mss
# Set the screen resolution (you can adjust this to your own screen resolution
def Screen_Shot(left=0, top=0, width=1920, height=1080):
	stc = mss.mss()
	scr = stc.grab({
		'left': left,
		'top': top,
		'width': width,
		'height': height
	})

	img = np.array(scr)
	img = cv2.cvtColor(img, cv2.IMREAD_COLOR)

	return img
count = 0
while True:
    count+=1
    image = Screen_Shot()
	# cv2_prac.re
    # cv2_prac.imshow("ga",np.copy(image))
    if count ==10:
        cv2.imwrite("haha.png",image)

    # Check for a key press and break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the window and close it
cv2.destroyAllWindows()
