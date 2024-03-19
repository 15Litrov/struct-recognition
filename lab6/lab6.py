import imgui
import OpenGL.GL as gl
import cv2
import os
import numpy as np

class Tracker:
    def __init__(self, tracker, name, color):
        self.builder = tracker
        self.name = name
        self.color = color
        self.recreate()

    def recreate(self):
        self.tracker = self.builder()

class Lab6Host:
    def __init__(self):
        self.openedTasks = set()
        self.frame_n = 0

        vidcap = cv2.VideoCapture('E:\STUDY\struct-recognition\data\\overtake.mp4')
        self.frames = []
        self.bbox = [478, 569, 87, 57]
        self.trackers = [
                         Tracker(cv2.TrackerKCF_create, "KCF", (0, 0, 255)),
                         Tracker(cv2.TrackerCSRT_create, "CSRT", (255, 0, 0)),
                         Tracker(cv2.TrackerMIL_create, "MIL", (255, 255, 0)),
                         ]
        self.use_tracker_ui = [True, True, True]
        self.use_tracker = [False, False, False]
        self.exec_time = [0, 0, 0]
        self.show_bbox = True
        self.pause = True

        self.display_tracker = [True, True, True]

        success,image = vidcap.read()
        while success:
            self.frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            success,image = vidcap.read()

        print(self.frames[0].shape)

    def update(self):
        if imgui.begin_main_menu_bar().opened:
            if imgui.menu_item("Track")[0]:
                self.openedTasks.add(self.task_Track)

            imgui.end_main_menu_bar()


        for task in list(self.openedTasks):
                task()

    def task_Track(self):
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        with imgui.begin("Track", True, flags) as window:
            if not window.opened or len(self.frames) == 0:
                self.openedTasks.remove(self.task_Track)
                return
            
            imgui.text_ansi("Use tracker:")
            for i in range(len(self.trackers)):
                _, self.use_tracker_ui[i] = imgui.checkbox(self.trackers[i].name + "##use", self.use_tracker_ui[i])

            if imgui.button("Run"):
                self.frame_n = 0
                self.use_tracker = list.copy(self.use_tracker_ui)

                for i in range(len(self.trackers)):
                    print(self.trackers[i].name)
                    if self.use_tracker[i]:
                        self.trackers[i].recreate()
                        self.trackers[i].tracker.init(self.frames[0], (self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]))

            imgui.text_ansi("Init bbox:")
            _, self.bbox[0] = imgui.input_int("x0", self.bbox[0])
            _, self.bbox[1] = imgui.input_int("y0", self.bbox[1])
            _, self.bbox[2] = imgui.input_int("width", self.bbox[2])
            _, self.bbox[3] = imgui.input_int("height", self.bbox[3])

            imgui.separator()

            _, self.show_bbox = imgui.checkbox("bbox", self.show_bbox)
            _, self.pause = imgui.checkbox("pause", self.pause)

            imgui.text_ansi("Display tracker:")
            for i in range(len(self.trackers)):
                _, self.display_tracker[i] = imgui.checkbox(self.trackers[i].name + "##display", self.display_tracker[i])

            if not hasattr(self, "render_target"):
                self.render_target = self.create_texture(np.zeros(self.frames[0].shape, dtype="uint8"))

            current_frame_orig = self.frames[int(self.frame_n)]
            current_frame = np.copy(current_frame_orig)

            if self.show_bbox:
                bbox = self.bbox
                x1, y1 = bbox[0], bbox[1]
                width, height = bbox[2], bbox[3]
                cv2.rectangle(current_frame, (x1, y1), (x1+width, y1+height), (0, 0, 0), 2)

            for i in range(len(self.trackers)):
                if not self.use_tracker[i]:
                    self.exec_time[i] = -1
                    continue

                ok, bbox = self.trackers[i].tracker.update(current_frame_orig)
                if not ok or not self.display_tracker[i]:
                    continue

                x1, y1 = bbox[0], bbox[1]
                width, height = bbox[2], bbox[3]
                cv2.rectangle(current_frame, (x1, y1), (x1+width, y1+height), self.trackers[i].color, 2)


            gl.glBindTexture(gl.GL_TEXTURE_2D, self.render_target[0])
            gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, current_frame.shape[1], current_frame.shape[0], 
                               gl.GL_RGB, gl.GL_UNSIGNED_BYTE, current_frame)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

            if not self.pause:
                self.frame_n = np.min([self.frame_n + 1, len(self.frames) - 1])

            flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
            with imgui.begin("Player", True, flags) as window:
                    max_dim = max(self.render_target[1], self.render_target[2])
                    width = int(768 * (self.render_target[1] / max_dim))
                    height = int(768 * (self.render_target[2] / max_dim))
                    imgui.image(self.render_target[0], width, height, (0, 0), (1, 1))

    def create_texture(self, texData):
        width = texData.shape[1]
        height = texData.shape[0]

        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height, 0, gl.GL_RGB,
                        gl.GL_UNSIGNED_BYTE, texData)
        
        return (texture, width, height)