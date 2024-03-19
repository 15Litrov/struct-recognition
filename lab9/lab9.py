import imgui
import OpenGL.GL as gl
from OpenGL.GL import shaders
import cv2
import os
import numpy as np

class Lab9Host:
    def __init__(self):
        self.openedTasks = set()

        self.dst_o = cv2.imread(r"E:\STUDY\struct-recognition\data\underwater.png")
        # self.dst_o = cv2.resize(self.dst_o, (1800, 1000))
        self.dst_o = self.add_alpha(cv2.cvtColor(self.dst_o, cv2.COLOR_BGR2RGB) / 255.0)

        self.mask = cv2.imread(r"E:\STUDY\struct-recognition\data\fish_mask.png") / 255.0
        # self.mask = cv2.resize(self.mask, (512, 512))
        self.mask = self.add_alpha(self.mask)

        self.src_o = cv2.imread(r"E:\STUDY\struct-recognition\data\fish_masked.png")
        self.src_o = cv2.cvtColor(self.src_o, cv2.COLOR_BGR2RGB) / 255.0
        # self.src_o = cv2.resize(self.src_o, (512, 512))
        self.src_o = self.add_alpha(self.src_o)

        self.size = (self.src_o.shape[1], self.src_o.shape[0])
        self.offset = (200, 250)

        self.dst_c = np.copy(self.dst_o)
        self.dst_c[self.offset[1]:self.offset[1]+self.src_o.shape[0], self.offset[0]:self.offset[0]+self.src_o.shape[1]] = self.src_o

    def update(self):
        if imgui.begin_main_menu_bar().opened:
            if imgui.menu_item("Blend")[0]:
                self.openedTasks.add(self.task_Blend)

            imgui.end_main_menu_bar()

        for task in list(self.openedTasks):
                task()

    def task_Blend(self):
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        with imgui.begin("Blend", True, flags) as window:
            if not window.opened:
                self.openedTasks.remove(self.task_Blend)
                self.inited = False
                return
            
            if not hasattr(self, "inited"):
                self.inited = True
                self.iters = 0
                kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) / 4.0
                B = cv2.filter2D(self.src_o, -1, kernel, borderType=cv2.BORDER_REFLECT)
                self.tex_mask = self.create_texture(self.mask)
                self.tex_src_res = self.create_texture(B)
                self.tex_out = self.create_texture(self.dst_o)
                content = open(r"E:\STUDY\struct-recognition\lab9\compute.glsl", "r").read()
                compute = gl.glCreateShader(gl.GL_COMPUTE_SHADER)
                gl.glShaderSource(compute, content)
                gl.glCompileShader(compute)

                program = gl.glCreateProgram()
                gl.glAttachShader(program, compute)
                gl.glLinkProgram(program)
                self.prog = program


            gl.glUseProgram(self.prog)
            gl.glUniform2iv(gl.glGetUniformLocation(self.prog, "offset"), 1, self.offset)
            gl.glUniform2iv(gl.glGetUniformLocation(self.prog, "size"), 1, self.size)
            gl.glBindImageTexture(0, self.tex_src_res[0], 0, gl.GL_FALSE, 0, gl.GL_READ_WRITE, gl.GL_RGBA32F)
            gl.glBindImageTexture(1, self.tex_mask[0], 0, gl.GL_FALSE, 0, gl.GL_READ_WRITE, gl.GL_RGBA32F)
            gl.glBindImageTexture(2, self.tex_out[0], 0, gl.GL_FALSE, 0, gl.GL_READ_WRITE, gl.GL_RGBA32F)

            if self.iters < 50000:
                for _ in range(100):
                    gl.glDispatchCompute(self.src_o.shape[1]//8 + 1, self.src_o.shape[0]//8, 1)
                    gl.glMemoryBarrier(gl.GL_ALL_BARRIER_BITS)
                    self.iters += 1
                

            if self.iters >= 50000:
                gl.glUniform2iv(gl.glGetUniformLocation(self.prog, "offset"), 1, (self.offset[0] - 80, self.offset[1] - 70))
                for _ in range(100):
                    gl.glDispatchCompute(self.src_o.shape[1]//8 + 1, self.src_o.shape[0]//8, 1)
                    gl.glMemoryBarrier(gl.GL_ALL_BARRIER_BITS)
                    self.iters += 1

            imgui.text_ansi(str(self.iters))
            imgui.image(self.tex_out[0], self.tex_out[1], self.tex_out[2], (0, 0), (1, 1))

    def add_alpha(self, img):
        dst = np.ones((img.shape[0], img.shape[1], 4))
        dst[:, :, :3] = img
        return dst

    def create_texture(self, texData):
        width = texData.shape[1]
        height = texData.shape[0]

        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA32F, width, height, 0, gl.GL_RGBA,
                        gl.GL_FLOAT, texData)
        
        return (texture, width, height)