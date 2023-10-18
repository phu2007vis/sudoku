from pynput.keyboard import Key ,Controller

import math
import torch


from solver import Solve_Main
import cv2
import numpy as np
import pyautogui
from capture import Capture
import win32api
import time
import win32con
import cv2
def click(x,y):
    win32api.SetCursorPos((x,y))
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0)
    time.sleep(0.01)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=7)
        self.batch1 = torch.nn.BatchNorm2d(10)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=9,bias = False)
        self.batch2 = torch.nn.BatchNorm2d(20)
        self.fc1 = torch.nn.Linear(980, 256)
        self.batch3 = torch.nn.BatchNorm1d(256)
        self.dropout1 = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(256,10 )
    def forward(self, x):
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv1(x), 2))
        x = self.batch1(x)
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv2(x), 2))
        x = self.batch2(x)
        x = x.view(-1, 980)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
def read_gray(path_image):
    if type(path_image) is str:
        image = cv2.imread(path_image)
    else:
        image = path_image
    origin_image = np.copy(image)

    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image,(3,3),5)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    gray_image = cv2.bitwise_not(image)

    return gray_image,origin_image
def find_board(image,origin_image):
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours,key = cv2.contourArea,reverse=True)
    board = None
    for contour in contours:
      if cv2.contourArea(contour) > image.shape[0] * image.shape[1]* 0.1:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 :
          board_cnf = approx
          board = np.array(approx).squeeze()
          break

    board = board[np.argsort(board[:,0])]
    lefts = board[:2]
    lefts = lefts[np.argsort(lefts[:,1])]
    tl,bl = lefts[0], lefts[1]
    rights = board[2:]
    rights = rights[np.argsort(rights[:,1])]
    tr,br = rights[0],rights[0]
    roi = image[tl[1]:bl[1],tl[0]:tr[0]]


    return board_cnf,roi,(tl,tr,bl,br)
def extract_cell(path_image,margin_vertical = 10,margin_horizal = 15):

    gray_image,origin_image = read_gray(path_image)
    board_cnf,roi,location = find_board(gray_image,origin_image)

    length = int(math.sqrt(cv2.contourArea(board_cnf)/82))
    count = 1
    cell_datas = []
    mask = []
    for i in range(9):
      for j in range(9):

        cell = roi[length*i+margin_vertical:length*(i+1)-margin_vertical,length*j+margin_horizal:length*(j+1)-margin_horizal]
        cell = cv2.resize(cell,(50,50))
        cell = cv2.erode(cell,(5,5),2)

        if np.sum(np.where(cell==0,0,1))>100:
            mask.append(False)
        else:
            mask.append(True)

        cell_datas.append(torch.tensor(cell/255.0,dtype = torch.float32).unsqueeze(0).unsqueeze(0))

        count+=1
    location = location + (length,)
    return cell_datas,mask,location,origin_image

class Solver:
    def __init__(self,model_path,device = 'cpu',margin_vertical = 10,margin_horizal = 15):

        self.model = torch.load(model_path,map_location=torch.device('cpu')).eval()

        self.device = device

        self.margin_vertical =  margin_vertical
        self.margin_horizal =   margin_horizal

        self.my_keyboard = Controller()

    def predict(self,data_cell,mask):
        data_cell = torch.cat(data_cell,0).to(self.device)

        value = self.model(data_cell)

        _ , number = value.max(1)
        number[mask] = 0
        return number

    def solve(self,image_path,cap:Capture,print = True):

        data_cell,mask,location,origin_image  = extract_cell(image_path)

        numbers = self.predict(data_cell,mask)

        self.sudoku = Solve_Main(numbers)

        ret,self.board = self.sudoku.solve()
        self.ret = ret
        self.cap = cap
        self.location = location

        if print:
            self.print_image(origin_image, location)
        else:
            self.press()
    def press(self):
        self.cap.window.activate()
        self.cap.window.moveTo(0,0)
        tl,bl,tr,br,length = self.location

        location = list.copy(self.cap.location)

        location[0] += tl[0]+20
        location[1]+=tl[1]+20
        click(location[0],location[1])

        direction = 0

        time_delay = 0.11   
        for i in range(9):
            for j in range(9) if i%2==0 else range(9)[::-1]:
                # Simulate pressing the current cell value

                self.my_keyboard.press(str(self.board[i*9+j]))
                time.sleep(time_delay)
                self.my_keyboard.release(str(self.board[i*9+j]))
                time.sleep(time_delay)
                if direction ==0:
                    if j<8:
                        self.my_keyboard.press(Key.right)
                        time.sleep(time_delay)
                        self.my_keyboard.release(Key.right)
                        time.sleep(time_delay)
                    else:
                        direction = 1
                        self.my_keyboard.press(Key.down)
                        time.sleep(time_delay)
                        self.my_keyboard.release(Key.down)
                        time.sleep(time_delay)
                else:
                    if j>0:
                        self.my_keyboard.press(Key.left)
                        time.sleep(time_delay)
                        self.my_keyboard.release(Key.left)
                        time.sleep(time_delay)
                    else:
                        direction=0
                        self.my_keyboard.press(Key.down)
                        time.sleep(time_delay)
                        self.my_keyboard.release(Key.down)
                        time.sleep(time_delay)



    def print_image(self,origin_image,location):

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.5
        font_color = (255,0,0)  # White color in BGR
        font_thickness = 4
        position = (20, 60)

        if not self.ret:
            cv2.putText(origin_image , "ngu lol", position, font, font_scale, font_color, font_thickness)
        else:

            tl,tr,bl,br,length = location
            xl,yl = tl[0],tl[1]

            for i in range(9):
                for j in range(9):

                    if self.sudoku.is_default[i*9+j]:
                        continue


                    y1 = length * i + self.margin_vertical+yl
                    y2 = length * (i + 1) - self.margin_vertical+yl
                    x1 = length * j + self.margin_horizal+xl
                    x2 = length * (j + 1) - self.margin_horizal+xl

                    text = str(self.board[i*9+j])
                    cv2.putText(origin_image[y1:y2,x1:x2] , text, position, font, font_scale, font_color, font_thickness)

            cv2.imshow("origin",origin_image)
            cv2.waitKey(30000)







solver = Solver(r"D:\python\sodoku\predict_number.pt")





import keyboard
from  capture import Capture

cap = Capture("Microsoft Sudoku")


def on_key_event():
            screenshot,ret = cap.take_screenshot()

            if ret:
                screenshot = cv2.cvtColor(screenshot,cv2.COLOR_BGR2RGB)
                solver.solve(screenshot,cap)





while True:
    if keyboard.is_pressed("q"):
        break
    elif keyboard.is_pressed("a"):
        on_key_event()

import time
import win32api
import win32con




