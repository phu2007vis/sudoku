import cv2
import numpy as np
import pyautogui
import pygetwindow
import keyboard
import win32gui
import pywinauto
class Capture:
    def __init__(self,window_title):
        self.window_title = window_title
        self.window = pygetwindow.getWindowsWithTitle(self.window_title)[0]
        self.location = [0,0]
    def active(self):
        pywinauto.application.Application().connect(handle=self.window._hWnd).top_window().set_focus()
    def take_screenshot(self):
        try:

            window = pygetwindow.getWindowsWithTitle(self.window_title)[0]


            window.moveTo(0,0)
            # self.window = window
            if not window.isActive:
                self.active()

            x, y, width, height = window.left, window.top, window.width, window.height


            # Get the border and title bar dimensions



            margin = [15,15,30,30] #top left width height
            self.location = [x+margin[0], y+margin[1]]
            # Capture the screenshot

            screenshot = pyautogui.screenshot(region=(x + margin[0], y + margin[1], width - margin[2], height - margin[3]))

            return cv2.cvtColor(np.array(screenshot),cv2.COLOR_BGR2RGB),True

        except IndexError:

            print("Widow not found.")
            return 1, False

        # except Exception as e:
        #     print(f"An error occurred: {str(e)}")
        #     exit()
    @staticmethod
    def enum_window_titles():
        def callback(handle, data):
            if win32gui.IsWindowVisible(handle) and win32gui.GetWindowText(handle) != "":
                print(win32gui.GetWindowText(handle))

        win32gui.EnumWindows(callback, None)






