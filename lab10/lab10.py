import imgui
import numpy as np
import pickle
import glfw

class Container:
    def __init__(self):
        pass

class Lab10Host:
    def __init__(self):
        self.openedTasks = set()
        self.password = "hellolab10"
        self.password_input = ""
        self.time = glfw.get_time()
        self.carret_pos = 0
        self.history = []
        self.times = np.empty(len(self.password))
        self.input_id = 0

    def update(self):
        if imgui.begin_main_menu_bar().opened:
            if imgui.menu_item("Train")[0]:
                self.openedTasks.add(self.task_Train)

            if imgui.menu_item("Auth")[0]:
                self.openedTasks.add(self.task_Auth)

            imgui.end_main_menu_bar()

        for task in list(self.openedTasks):
                task()

    def task_Train(self):
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        with imgui.begin("Train", True, flags) as window:
            if not window.opened:
                self.openedTasks.remove(self.task_Train)
                self.inited = False
                self.password_input = ""
                self.carret_pos = 0
                self.times = np.empty(len(self.password))
                return
            
            if self.carret_pos != 0:
                imgui.text_ansi(self.password[:self.carret_pos])
                imgui.same_line()

            imgui.text_ansi_colored(self.password[self.carret_pos], 0, 1, 0)

            if self.carret_pos != len(self.password) - 1:
                imgui.same_line()
                imgui.text_ansi(self.password[self.carret_pos + 1:])

            _, _ = imgui.input_text(f"Password##train{self.input_id}", self.password_input, -1, imgui.INPUT_TEXT_PASSWORD | imgui.INPUT_TEXT_CALLBACK_CHAR_FILTER, self.callback_train_input)

            for h in self.history:
                imgui.text_ansi(str(h))

    def task_Auth(self):
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        with imgui.begin("Train", True, flags) as window:
            if not window.opened:
                self.openedTasks.remove(self.task_Auth)
                self.inited = False
                return
            
            if self.carret_pos != 0:
                imgui.text_ansi(self.password[:self.carret_pos])
                imgui.same_line()

            imgui.text_ansi_colored(self.password[self.carret_pos], 0, 1, 0)

            if self.carret_pos != len(self.password) - 1:
                imgui.same_line()
                imgui.text_ansi(self.password[self.carret_pos + 1:])

            _, _ = imgui.input_text(f"Password##auth{self.input_id}", self.password_input, -1, imgui.INPUT_TEXT_PASSWORD | imgui.INPUT_TEXT_CALLBACK_CHAR_FILTER, self.callback_auth_input)
    
    def callback_train_input(self, data):
        if data.event_char != self.password[self.carret_pos]:
            return 1
        
        t = glfw.get_time()
        dt = t - self.time
        self.time = t

        self.times[self.carret_pos] = dt
        self.carret_pos += 1

        if self.carret_pos == len(self.password):
            self.history.append(self.times[1:])
            self.times = np.empty(len(self.password))
            self.carret_pos = 0
            self.password_input = ""
            self.input_id += 1
            return 1

        return 0

    def callback_auth_input(self, data):
        if data.event_char != self.password[self.carret_pos]:
            return 1
        
        t = glfw.get_time()
        dt = t - self.time
        self.time = t

        self.times[self.carret_pos] = dt
        self.carret_pos += 1

        if self.carret_pos == len(self.password):
            isAuthorized(self.times)

            self.history.append(self.times)
            self.times = np.empty(len(self.password))
            self.carret_pos = 0
            self.password_input = ""
            self.input_id += 1
            return 1

        return 0
    
    def isAuthorized(self, times):
        