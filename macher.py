# import cv2
#
# from capture import Capture
# #
# # class Matcher:
# #     def __init__(self,image):
# #         self.Capture =
# cap = Capture("Microsoft sudoku")
# img =cap.take_screenshot()[0]
# gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# image_path = "image_so/so.png"
# temp = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
#
# h,w = temp.shape[:2]
#
#
# res = cv2.matchTemplate(gray,temp,cv2.TM_CCOEFF_NORMED)
# min_loc = cv2.minMaxLoc(res)[3]
# print(min_loc)
# cv2.rectangle(gray,min_loc,(min_loc[0]+w,min_loc[1]+h),0,2)
# cv2.imshow("hii",gray)
# cv2.waitKey(5000)
