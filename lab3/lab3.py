import imgui
import OpenGL.GL as gl
import cv2
import os
import numpy as np

rootPath = rf"E:\STUDY\struct-recognition\lab3"

class Lab3Host:
    def __init__(self):
        self.openedTasks = set()
        
        self.image1_path = "E:\STUDY\struct-recognition\data\kodim07.png"
        self.image2_path = "E:\STUDY\struct-recognition\data\watermark.png"

    def update(self):
            if imgui.begin_main_menu_bar().opened:
                if imgui.menu_item("Load images")[0]:
                    self.openedTasks.add(self.load_image_window)

                if imgui.begin_menu("Tasks").opened:
                    if imgui.menu_item("Invert")[0]:
                        self.openedTasks.add(self.task_invert)

                    if imgui.menu_item("Add const")[0]:
                        self.openedTasks.add(self.task_addconst)

                    if imgui.menu_item("Split")[0]:
                        self.openedTasks.add(self.task_split)

                    if imgui.menu_item("Blend")[0]:
                        self.openedTasks.add(self.task_blend)

                    if imgui.menu_item("Blur")[0]:
                        self.openedTasks.add(self.task_blur)

                    if imgui.menu_item("Sharpness")[0]:
                        self.openedTasks.add(self.task_sharpness)

                    if imgui.menu_item("Sobel")[0]:
                        self.openedTasks.add(self.task_sobel)

                    if imgui.menu_item("Median")[0]:
                        self.openedTasks.add(self.task_median)

                    if imgui.menu_item("Erosion")[0]:
                        self.openedTasks.add(self.task_erosion)

                    if imgui.menu_item("Dilatation")[0]:
                        self.openedTasks.add(self.task_dilatation)

                    if imgui.menu_item("Bit planes")[0]:
                        self.openedTasks.add(self.task_showplanes)

                    if imgui.menu_item("Embed watermark")[0]:
                        self.openedTasks.add(self.task_embedwatermark)

                    imgui.end_menu()

                imgui.end_main_menu_bar()

            for task in list(self.openedTasks):
                task()

    def load_image_window(self):
        with imgui.begin("Load images", True) as window:
            if not window.opened:
                self.openedTasks.remove(self.load_image_window)
                return

            changed1, text_val_1 = imgui.input_text("Image 1 path", self.image1_path)
            if changed1:
                self.image1_path = text_val_1

            changed2, text_val_2 = imgui.input_text("Image 2 path", self.image2_path)
            if changed2:
                self.image2_path = text_val_2

    def task_invert(self):
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        with imgui.begin("Invert", True, flags) as window:
            if not window.opened:
                self.openedTasks.remove(self.task_invert)

                # delete dexture
                if hasattr(self, 'test_image') and not self.test_image is None:
                     gl.glDeleteTextures([self.test_image[0]])
                     self.test_image = None

                return
            
            if imgui.button("Apply") and os.path.exists(self.image1_path):
                try:
                    img = cv2.imread(os.path.join(self.image1_path))
                    img = 255 - cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.test_image = self.create_texture(img)
                except:
                    self.test_image = None

            if not hasattr(self, 'test_image') or self.test_image is None:
                imgui.text("Please set image path in Load images tab and click Apply in this tab")
            else:
                max_dim = max(self.test_image[1], self.test_image[2])
                width = int(384 * (self.test_image[1] / max_dim))
                height = int(384 * (self.test_image[2] / max_dim))
                imgui.image(self.test_image[0], width, height, (0, 0), (1, 1))


    def task_addconst(self):
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        with imgui.begin("Add Const", True, flags) as window:
            if not window.opened:
                self.openedTasks.remove(self.task_addconst)

                # delete dexture
                if hasattr(self, 'image_addconst') and not self.image_addconst is None:
                     gl.glDeleteTextures([self.image_addconst[0]])
                     self.image_addconst = None

                return
            
            if not hasattr(self, "addconst_val"):
                self.addconst_val = [0, 0, 0]

            changed, vec3 = imgui.input_int3("add RGB:", *self.addconst_val)
            if changed:
                self.addconst_val = np.clip(vec3, -255, 255)

            if imgui.button("Apply") and os.path.exists(self.image1_path):
                try:
                    img = cv2.imread(os.path.join(self.image1_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype("int32")
                    img[:, :, 0] += self.addconst_val[0]
                    img[:, :, 1] += self.addconst_val[1]
                    img[:, :, 2] += self.addconst_val[2]
                    img = np.clip(img, 0, 255).astype("uint8")
                    self.image_addconst = self.create_texture(img)
                except:
                    self.image_addconst = None

            if not hasattr(self, 'image_addconst') or self.image_addconst is None:
                imgui.text("Please set image path in Load images tab and click Apply in this tab")
            else:
                max_dim = max(self.image_addconst[1], self.image_addconst[2])
                width = int(384 * (self.image_addconst[1] / max_dim))
                height = int(384 * (self.image_addconst[2] / max_dim))
                imgui.image(self.image_addconst[0], width, height, (0, 0), (1, 1))

    def task_split(self):
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        with imgui.begin("Split", True, flags) as window:
            if not window.opened:
                self.openedTasks.remove(self.task_split)

                # delete textures
                if hasattr(self, 'split_img_img') and not self.split_img_img is None:
                    gl.glDeleteTextures([self.split_img_img[0]])
                    gl.glDeleteTextures([self.split_img_r[0]])
                    gl.glDeleteTextures([self.split_img_g[0]])
                    gl.glDeleteTextures([self.split_img_b[0]])
                    self.split_img_img = None
                    self.split_img_r = None
                    self.split_img_g = None
                    self.split_img_b = None

                return
            
            if imgui.button("Apply") and os.path.exists(self.image1_path):
                try:
                    img = cv2.imread(os.path.join(self.image1_path))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_r = img.copy()
                    img_r[:, :, 1] = 0
                    img_r[:, :, 2] = 0
                    img_g = img.copy()
                    img_g[:, :, 0] = 0
                    img_g[:, :, 2] = 0
                    img_b = img.copy()
                    img_b[:, :, 0] = 0
                    img_b[:, :, 1] = 0
                    self.split_img_img = self.create_texture(img)
                    self.split_img_r = self.create_texture(img_r)
                    self.split_img_g = self.create_texture(img_g)
                    self.split_img_b = self.create_texture(img_b)
                except:
                    self.split_img_img = None
                    self.split_img_r = None
                    self.split_img_g = None
                    self.split_img_b = None

            if not hasattr(self, 'split_img_img') or self.split_img_img is None:
                imgui.text("Please set image path in Load images tab and click Apply in this tab")
            else:
                max_dim = max(self.split_img_img[1], self.split_img_img[2])
                width = int(384 * (self.split_img_img[1] / max_dim))
                height = int(384 * (self.split_img_img[2] / max_dim))
                imgui.image(self.split_img_img[0], width, height, (0, 0), (1, 1))
                imgui.same_line()
                imgui.image(self.split_img_r[0], width, height, (0, 0), (1, 1))
                imgui.image(self.split_img_g[0], width, height, (0, 0), (1, 1))
                imgui.same_line()
                imgui.image(self.split_img_b[0], width, height, (0, 0), (1, 1))
        
    def task_blend(self):
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        with imgui.begin("Blend images", True, flags) as window:
            if not window.opened:
                self.openedTasks.remove(self.task_blend)

                # delete dexture
                if hasattr(self, 'image_blended') and not self.image_blended is None:
                     gl.glDeleteTextures([self.image_blended[0]])
                     self.image_blended = None

                return
            
            if not hasattr(self, "blend_alpha_step"):
                self.blend_alpha_step = 0.125

            if not hasattr(self, "blend_alpha"):
                self.blend_alpha = 0

            if not hasattr(self, "blend_timer"):
                self.blend_timer = 0

            self.blend_timer += imgui.get_io().delta_time
            if (self.blend_timer > 0.5):
                self.blend_timer = 0
                self.blend_alpha = (self.blend_alpha + self.blend_alpha_step) % 1
            
                if os.path.exists(self.image1_path) and os.path.exists(self.image2_path):
                    img1 = cv2.imread(os.path.join(self.image1_path))
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) / 255.0
                    img2 = cv2.imread(os.path.join(self.image2_path))
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) / 255.0
                    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

                    blended = self.blend_alpha * img1 + (1 - self.blend_alpha) * img2
                    blended = (np.clip(blended, 0, 1) * 255).astype("uint8")

                    if hasattr(self, 'image_blended') and not self.image_blended is None:
                        gl.glDeleteTextures([self.image_blended[0]])

                    self.image_blended = self.create_texture(blended)

            imgui.text_ansi(f"Alpha: {self.blend_alpha}")

            changed, step = imgui.input_float("Alpha step", self.blend_alpha_step)
            if changed:
                self.blend_alpha_step = step

            if not hasattr(self, 'image_blended') or self.image_blended is None:
                imgui.text("Please set image path in Load images tab")
            else:
                max_dim = max(self.image_blended[1], self.image_blended[2])
                width = int(384 * (self.image_blended[1] / max_dim))
                height = int(384 * (self.image_blended[2] / max_dim))
                imgui.image(self.image_blended[0], width, height, (0, 0), (1, 1))

    def calc_window(self, data, kernel, func = np.sum):
        ker_size = np.array([kernel.shape[0] // 2, kernel.shape[1] // 2])

        padded = np.zeros(data.shape + 2 * ker_size)
        padded[ker_size[0]:ker_size[0] + data.shape[0], ker_size[1]:ker_size[1] + data.shape[1]] = data

        output = np.empty(data.shape)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                output[i, j] = func(padded[i:(i + kernel.shape[0]), j:(j + kernel.shape[1])] * kernel)

        return output

    def task_blur(self):
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        with imgui.begin("Blur", True, flags) as window:
            if not window.opened:
                self.openedTasks.remove(self.task_blur)

                # delete dexture
                if hasattr(self, 'blurred_cv2') and not self.blurred_cv2 is None:
                     gl.glDeleteTextures([self.blurred_cv2[0]])
                     self.blurred_cv2 = None

                if hasattr(self, 'blurred_my') and not self.blurred_my is None:
                     gl.glDeleteTextures([self.blurred_my[0]])
                     self.blurred_my = None

                if hasattr(self, 'blurred_diff') and not self.blurred_diff is None:
                     gl.glDeleteTextures([self.blurred_diff[0]])
                     self.blurred_diff = None

                if hasattr(self, 'blurred_original') and not self.blurred_original is None:
                     gl.glDeleteTextures([self.blurred_original[0]])
                     self.blurred_original = None

                return
            
            if imgui.button("Apply") and os.path.exists(self.image1_path):
                try:
                    img = cv2.imread(os.path.join(self.image1_path))
                    img_0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img_0/ 255.0

                    kernel = np.ones((5, 5)) / 25.0
                    output = np.empty(img.shape)
                    for i in range(3):
                        output[:, :, i] = self.calc_window(img[:, :, i], kernel)
                    output = np.clip((output * 255), 0, 255).astype("uint8")

                    output_cv2 = cv2.filter2D(img_0, -1, kernel, borderType=cv2.BORDER_DEFAULT)

                    diff = np.abs(output.astype("int16") - output_cv2.astype("int16")).astype("uint8")

                    self.blurred_original = self.create_texture(img_0)
                    self.blurred_my = self.create_texture(output)
                    self.blurred_cv2 = self.create_texture(output_cv2)
                    self.blurred_diff = self.create_texture(diff)
                except:
                    self.blurred_my = None
                    self.blurred_original = None
                    self.blurred_cv2 = None
                    self.blurred_diff = None

            if not hasattr(self, 'blurred_my') or self.blurred_my is None:
                imgui.text("Please set image path in Load images tab and click Apply in this tab")
            else:
                max_dim = max(self.blurred_my[1], self.blurred_my[2])
                width = int(384 * (self.blurred_my[1] / max_dim))
                height = int(384 * (self.blurred_my[2] / max_dim))

                imgui.image(self.blurred_my[0], width, height, (0, 0), (1, 1))
                imgui.same_line()
                imgui.image(self.blurred_cv2[0], width, height, (0, 0), (1, 1))
                imgui.image(self.blurred_original[0], width, height, (0, 0), (1, 1))
                imgui.same_line()
                imgui.image(self.blurred_diff[0], width, height, (0, 0), (1, 1))

    def task_sharpness(self):
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        with imgui.begin("Sharpness", True, flags) as window:
            if not window.opened:
                self.openedTasks.remove(self.task_sharpness)

                # delete dexture
                if hasattr(self, 'sharpened_cv2') and not self.sharpened_cv2 is None:
                     gl.glDeleteTextures([self.sharpened_cv2[0]])
                     self.sharpened_cv2 = None

                if hasattr(self, 'sharpened_my') and not self.sharpened_my is None:
                     gl.glDeleteTextures([self.sharpened_my[0]])
                     self.sharpened_my = None

                if hasattr(self, 'sharpened_diff') and not self.sharpened_diff is None:
                     gl.glDeleteTextures([self.sharpened_diff[0]])
                     self.sharpened_diff = None

                if hasattr(self, 'sharpened_original') and not self.sharpened_original is None:
                     gl.glDeleteTextures([self.sharpened_original[0]])
                     self.sharpened_original = None

                return
            
            if imgui.button("Apply") and os.path.exists(self.image1_path):
                try:
                    img = cv2.imread(os.path.join(self.image1_path))
                    img_0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img_0/ 255.0

                    kernel = np.full((3, 3), -1)
                    kernel[1, 1] = 9

                    output = np.empty(img.shape)
                    for i in range(3):
                        output[:, :, i] = self.calc_window(img[:, :, i], kernel)
                    output = np.clip((output * 255), 0, 255).astype("uint8")

                    output_cv2 = cv2.filter2D(img_0, -1, kernel, borderType=cv2.BORDER_DEFAULT)

                    diff = np.abs(output.astype("int16") - output_cv2.astype("int16")).astype("uint8")

                    self.sharpened_original = self.create_texture(img_0)
                    self.sharpened_my = self.create_texture(output)
                    self.sharpened_cv2 = self.create_texture(output_cv2)
                    self.sharpened_diff = self.create_texture(diff)
                except:
                    self.sharpened_my = None
                    self.sharpened_original = None
                    self.sharpened_cv2 = None
                    self.sharpened_diff = None

            if not hasattr(self, 'sharpened_my') or self.sharpened_my is None:
                imgui.text("Please set image path in Load images tab and click Apply in this tab")
            else:
                max_dim = max(self.sharpened_my[1], self.sharpened_my[2])
                width = int(384 * (self.sharpened_my[1] / max_dim))
                height = int(384 * (self.sharpened_my[2] / max_dim))

                imgui.image(self.sharpened_my[0], width, height, (0, 0), (1, 1))
                imgui.same_line()
                imgui.image(self.sharpened_cv2[0], width, height, (0, 0), (1, 1))
                imgui.image(self.sharpened_original[0], width, height, (0, 0), (1, 1))
                imgui.same_line()
                imgui.image(self.sharpened_diff[0], width, height, (0, 0), (1, 1))

    def task_sobel(self):
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        with imgui.begin("Sobel", True, flags) as window:
            if not window.opened:
                self.openedTasks.remove(self.task_sobel)

                # delete dexture
                if hasattr(self, 'sobel_cv2') and not self.sobel_cv2 is None:
                     gl.glDeleteTextures([self.sobel_cv2[0]])
                     self.sobel_cv2 = None

                if hasattr(self, 'sobel_my') and not self.sobel_my is None:
                     gl.glDeleteTextures([self.sobel_my[0]])
                     self.sobel_my = None

                if hasattr(self, 'sobel_diff') and not self.sobel_diff is None:
                     gl.glDeleteTextures([self.sobel_diff[0]])
                     self.sobel_diff = None

                if hasattr(self, 'sobel_original') and not self.sobel_original is None:
                     gl.glDeleteTextures([self.sobel_original[0]])
                     self.sobel_original = None

                return
            
            if imgui.button("Apply") and os.path.exists(self.image1_path):
                try:
                    img = cv2.imread(os.path.join(self.image1_path))
                    img_0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = img_0 / 255.0

                    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

                    output_x = self.calc_window(img, kernel_x)
                    output_y = self.calc_window(img, kernel_y)
                    output = np.sqrt(output_x**2 + output_y**2)
                    output = np.clip((output * 255), 0, 255).astype("uint8")

                    cv2_output_x = cv2.Sobel(img, -1, 1, 0, ksize=3, scale=1, borderType=cv2.BORDER_DEFAULT)
                    cv2_output_y = cv2.Sobel(img, -1, 0, 1, ksize=3, scale=1, borderType=cv2.BORDER_DEFAULT)

                    cv2_output = np.sqrt(cv2_output_x**2 + cv2_output_y**2)
                    cv2_output = np.clip((cv2_output * 255), 0, 255).astype("uint8")

                    diff = np.abs(output.astype("int16") - cv2_output.astype("int16")).astype("uint8")

                    self.sobel_original = self.create_texture(cv2.merge([img_0, img_0, img_0]))
                    self.sobel_my = self.create_texture(cv2.merge([output, output, output]))
                    self.sobel_cv2 = self.create_texture(cv2.merge([cv2_output, cv2_output, cv2_output]))
                    self.sobel_diff = self.create_texture(cv2.merge([diff, diff, diff]))
                except:
                    self.sobel_my = None
                    self.sobel_original = None
                    self.sobel_cv2 = None
                    self.sobel_diff = None

            if not hasattr(self, 'sobel_my') or self.sobel_my is None:
                imgui.text("Please set image path in Load images tab and click Apply in this tab")
            else:
                max_dim = max(self.sobel_my[1], self.sobel_my[2])
                width = int(384 * (self.sobel_my[1] / max_dim))
                height = int(384 * (self.sobel_my[2] / max_dim))

                imgui.image(self.sobel_my[0], width, height, (0, 0), (1, 1))
                imgui.same_line()
                imgui.image(self.sobel_cv2[0], width, height, (0, 0), (1, 1))

                imgui.image(self.sobel_original[0], width, height, (0, 0), (1, 1))
                imgui.same_line()
                imgui.image(self.sobel_diff[0], width, height, (0, 0), (1, 1))

    def task_median(self):
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        with imgui.begin("Median", True, flags) as window:
            if not window.opened:
                self.openedTasks.remove(self.task_median)

                # delete dexture
                if hasattr(self, 'median_cv2') and not self.median_cv2 is None:
                     gl.glDeleteTextures([self.median_cv2[0]])
                     self.median_cv2 = None

                if hasattr(self, 'median_my') and not self.median_my is None:
                     gl.glDeleteTextures([self.median_my[0]])
                     self.median_my = None

                if hasattr(self, 'median_diff') and not self.median_diff is None:
                     gl.glDeleteTextures([self.median_diff[0]])
                     self.median_diff = None

                if hasattr(self, 'median_original') and not self.median_original is None:
                     gl.glDeleteTextures([self.median_original[0]])
                     self.median_original = None

                return
            
            if imgui.button("Apply") and os.path.exists(self.image1_path):
                try:
                    img = cv2.imread(os.path.join(self.image1_path))
                    img_0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img_0/ 255.0

                    kernel = np.ones((5, 5))
                    output = np.empty(img.shape)
                    for i in range(3):
                        output[:, :, i] = self.calc_window(img[:, :, i], kernel, np.median)
                    output = np.clip((output * 255), 0, 255).astype("uint8")

                    output_cv2 = cv2.medianBlur(img_0, 5)

                    diff = np.abs(output.astype("int16") - output_cv2.astype("int16")).astype("uint8")

                    self.median_original = self.create_texture(img_0)
                    self.median_my = self.create_texture(output)
                    self.median_cv2 = self.create_texture(output_cv2)
                    self.median_diff = self.create_texture(diff)
                except:
                    self.median_my = None
                    self.median_original = None
                    self.median_cv2 = None
                    self.median_diff = None

            if not hasattr(self, 'median_my') or self.median_my is None:
                imgui.text("Please set image path in Load images tab and click Apply in this tab")
            else:
                max_dim = max(self.median_my[1], self.median_my[2])
                width = int(384 * (self.median_my[1] / max_dim))
                height = int(384 * (self.median_my[2] / max_dim))

                imgui.image(self.median_my[0], width, height, (0, 0), (1, 1))
                imgui.same_line()
                imgui.image(self.median_cv2[0], width, height, (0, 0), (1, 1))
                imgui.image(self.median_original[0], width, height, (0, 0), (1, 1))
                imgui.same_line()
                imgui.image(self.median_diff[0], width, height, (0, 0), (1, 1))

    def task_erosion(self):
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        with imgui.begin("Erosion", True, flags) as window:
            if not window.opened:
                self.openedTasks.remove(self.task_erosion)

                # delete dexture
                if hasattr(self, 'erosion_cv2') and not self.erosion_cv2 is None:
                     gl.glDeleteTextures([self.erosion_cv2[0]])
                     self.erosion_cv2 = None

                if hasattr(self, 'erosion_my') and not self.erosion_my is None:
                     gl.glDeleteTextures([self.erosion_my[0]])
                     self.erosion_my = None

                if hasattr(self, 'erosion_diff') and not self.erosion_diff is None:
                     gl.glDeleteTextures([self.erosion_diff[0]])
                     self.erosion_diff = None

                if hasattr(self, 'erosion_original') and not self.erosion_original is None:
                     gl.glDeleteTextures([self.erosion_original[0]])
                     self.erosion_original = None

                return
            
            if imgui.button("Apply") and os.path.exists(self.image1_path):
                try:
                    img = cv2.imread(os.path.join(self.image1_path))
                    img_0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img_0/ 255.0

                    kernel = np.ones((5, 5))
                    output = np.empty(img.shape)
                    for i in range(3):
                        output[:, :, i] = self.calc_window(img[:, :, i], kernel, np.min)
                    output = np.clip((output * 255), 0, 255).astype("uint8")

                    output_cv2 = cv2.erode(img_0, kernel, borderType=cv2.BORDER_DEFAULT)

                    diff = np.abs(output.astype("int16") - output_cv2.astype("int16")).astype("uint8")

                    self.erosion_original = self.create_texture(img_0)
                    self.erosion_my = self.create_texture(output)
                    self.erosion_cv2 = self.create_texture(output_cv2)
                    self.erosion_diff = self.create_texture(diff)
                except:
                    self.erosion_my = None
                    self.erosion_original = None
                    self.erosion_cv2 = None
                    self.erosion_diff = None

            if not hasattr(self, 'erosion_my') or self.erosion_my is None:
                imgui.text("Please set image path in Load images tab and click Apply in this tab")
            else:
                max_dim = max(self.erosion_my[1], self.erosion_my[2])
                width = int(384 * (self.erosion_my[1] / max_dim))
                height = int(384 * (self.erosion_my[2] / max_dim))

                imgui.image(self.erosion_my[0], width, height, (0, 0), (1, 1))
                imgui.same_line()
                imgui.image(self.erosion_cv2[0], width, height, (0, 0), (1, 1))
                imgui.image(self.erosion_original[0], width, height, (0, 0), (1, 1))
                imgui.same_line()
                imgui.image(self.erosion_diff[0], width, height, (0, 0), (1, 1))

    def task_dilatation(self):
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        with imgui.begin("dilatation", True, flags) as window:
            if not window.opened:
                self.openedTasks.remove(self.task_dilatation)

                # delete dexture
                if hasattr(self, 'dilatation_cv2') and not self.dilatation_cv2 is None:
                     gl.glDeleteTextures([self.dilatation_cv2[0]])
                     self.dilatation_cv2 = None

                if hasattr(self, 'dilatation_my') and not self.dilatation_my is None:
                     gl.glDeleteTextures([self.dilatation_my[0]])
                     self.dilatation_my = None

                if hasattr(self, 'dilatation_diff') and not self.dilatation_diff is None:
                     gl.glDeleteTextures([self.dilatation_diff[0]])
                     self.dilatation_diff = None

                if hasattr(self, 'dilatation_original') and not self.dilatation_original is None:
                     gl.glDeleteTextures([self.dilatation_original[0]])
                     self.dilatation_original = None

                return
            
            if imgui.button("Apply") and os.path.exists(self.image1_path):
                try:
                    img = cv2.imread(os.path.join(self.image1_path))
                    img_0 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = img_0/ 255.0

                    kernel = np.ones((5, 5))
                    output = np.empty(img.shape)
                    for i in range(3):
                        output[:, :, i] = self.calc_window(img[:, :, i], kernel, np.max)
                    output = np.clip((output * 255), 0, 255).astype("uint8")

                    output_cv2 = cv2.dilate(img_0, kernel)

                    diff = np.abs(output.astype("int16") - output_cv2.astype("int16")).astype("uint8")

                    self.dilatation_original = self.create_texture(img_0)
                    self.dilatation_my = self.create_texture(output)
                    self.dilatation_cv2 = self.create_texture(output_cv2)
                    self.dilatation_diff = self.create_texture(diff)
                except:
                    self.dilatation_my = None
                    self.dilatation_original = None
                    self.dilatation_cv2 = None
                    self.dilatation_diff = None

            if not hasattr(self, 'dilatation_my') or self.dilatation_my is None:
                imgui.text("Please set image path in Load images tab and click Apply in this tab")
            else:
                max_dim = max(self.dilatation_my[1], self.dilatation_my[2])
                width = int(384 * (self.dilatation_my[1] / max_dim))
                height = int(384 * (self.dilatation_my[2] / max_dim))

                imgui.image(self.dilatation_my[0], width, height, (0, 0), (1, 1))
                imgui.same_line()
                imgui.image(self.dilatation_cv2[0], width, height, (0, 0), (1, 1))
                imgui.image(self.dilatation_original[0], width, height, (0, 0), (1, 1))
                imgui.same_line()
                imgui.image(self.dilatation_diff[0], width, height, (0, 0), (1, 1))

    def get_plane(self, data, plane):
        shifts = plane - 1
        # make zero all except target plane AND shift plane to least bit
        return (data & (1 << shifts)) >> shifts
    
    def set_plane(self, dst, src, plane):
        shifts = plane - 1
        # keep original values of all bits except target, which sets to 0
        dst1 = dst & (~(1 << shifts))
        # set to 0 all planes expet target
        src1 = src & (1 << shifts)
        return dst1 | src1
    
    def xor_plane(self, dst, src, plane):
        shifts = plane - 1
        src_1 = src & (1 << shifts)
        dst_1 = dst & (1 << shifts)
        xor = dst_1 ^ src_1
        dst2 = dst & (~(1 << shifts))
        return dst2 | xor



    def task_showplanes(self):
        with imgui.begin("Bit planes", True) as window:
            if not window.opened:
                self.openedTasks.remove(self.task_showplanes)

                if hasattr(self, 'images_planes') and not self.images_planes is None:
                    for i in range(9):
                        gl.glDeleteTextures([self.images_planes[i][0]])
                    self.images_planes = None

            if imgui.button("Apply") and os.path.exists(self.image1_path):
                img = cv2.imread(os.path.join(self.image1_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                img_arr = [None]*9
                self.images_planes = img_arr

                img_arr[0]= self.create_texture(img)
                for i in range(8):
                    img_arr[i + 1] = self.create_texture(self.get_plane(img, i + 1) * 255)

            if not hasattr(self, 'images_planes') or self.images_planes is None:
                imgui.text("Please set image path in Load images tab and click Apply in this tab")
            else:
                max_dim = max(self.images_planes[0][1], self.images_planes[0][2])
                width = int(384 * (self.images_planes[0][1] / max_dim))
                height = int(384 * (self.images_planes[0][2] / max_dim))

                imgui.image(self.images_planes[0][0], width, height, (0, 0), (1, 1))
                for i in range(4):
                    imgui.image(self.images_planes[1 + 2*i][0], width, height, (0, 0), (1, 1))
                    imgui.same_line()
                    imgui.image(self.images_planes[2 + 2*i][0], width, height, (0, 0), (1, 1))

    def task_embedwatermark(self):
        with imgui.begin("Embed watermark", True) as window:
            if not window.opened:
                self.openedTasks.remove(self.task_embedwatermark)

                if hasattr(self, 'images_embedded') and not self.images_embedded is None:
                    for i in range(4):
                        gl.glDeleteTextures([self.images_embedded[i][0]])
                    self.images_embedded = None
            
            if not hasattr(self, 'embed_channel') or not hasattr(self, 'embed_plane') or not hasattr(self, 'embed_xor') or not hasattr(self, 'embed_save_path'):
                self.embed_channel = 0
                self.embed_plane = 1
                self.embed_xor = False
                self.embed_save_path = ""

            changed_channel, channel_val = imgui.input_int("Target channel (0 - R, 1 - G, 2 - B)", self.embed_channel)
            changed_plane, plane_val = imgui.input_int("Target bit plane [1, 8]", self.embed_plane)
            _, self.embed_xor = imgui.checkbox("Use XOR", self.embed_xor)
            changed_path, path = imgui.input_text("Save path", self.embed_save_path)

            if changed_channel:
                self.embed_channel = np.clip(channel_val, 0, 2)
            
            if changed_plane:
                self.embed_plane = np.clip(plane_val, 1, 8)

            if changed_path:
                self.embed_save_path = path

            if imgui.button("Apply") and os.path.exists(self.image1_path) and os.path.exists(self.image2_path):
                dst_img = cv2.imread(os.path.join(self.image1_path))
                dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)
                src_img = cv2.imread(os.path.join(self.image2_path))
                src_img = ((cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY) > 127).astype("uint8") * 255).astype("uint8")
                
                img_arr = [None]*4
                self.images_embedded = img_arr

                img_arr[0] = self.create_texture(dst_img)
                img_arr[1] = self.create_texture(cv2.merge([src_img, src_img, src_img]))

                src_img = self.repeat_resize(src_img, dst_img.shape[:2])
                img_arr[2] = self.create_texture(cv2.merge([src_img, src_img, src_img]))

                output = dst_img
                if not self.embed_xor:
                    output[:, :, self.embed_channel] = self.set_plane(dst_img[:, :, self.embed_channel], src_img, self.embed_plane)
                else:
                    output[:, :, self.embed_channel] = self.xor_plane(dst_img[:, :, self.embed_channel], src_img, self.embed_plane)

                img_arr[3] = self.create_texture(output)

                self.watermarked_image = output

            imgui.same_line()

            if imgui.button("Save") and hasattr(self, 'embed_save_path') and hasattr(self, "watermarked_image") and not self.watermarked_image is None:
                try:
                    o = cv2.cvtColor(self.watermarked_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(self.embed_save_path, o)
                except:
                    print("failed to save")

            if not hasattr(self, 'images_embedded') or self.images_embedded is None:
                imgui.text("Please set image path in Load images tab and click Apply in this tab")
            else:
                max_dim = max(self.images_embedded[0][1], self.images_embedded[0][2])
                width = int(384 * (self.images_embedded[0][1] / max_dim))
                height = int(384 * (self.images_embedded[0][2] / max_dim))

                imgui.image(self.images_embedded[0][0], width, height, (0, 0), (1, 1))
                imgui.same_line()
                imgui.image(self.images_embedded[1][0], width, height, (0, 0), (1, 1))
                
                imgui.image(self.images_embedded[2][0], width, height, (0, 0), (1, 1))
                imgui.same_line()
                imgui.image(self.images_embedded[3][0], width, height, (0, 0), (1, 1))


    def repeat_resize(self, src, dst_shape):
        dst = np.empty(dst_shape, dtype=src.dtype)
        r_i = dst_shape[0] // src.shape[0]
        r_j = dst_shape[1] // src.shape[1]
        e_i = dst_shape[0] % src.shape[0]
        e_j = dst_shape[1] % src.shape[1]

        si = min(src.shape[0], dst.shape[0])
        sj = min(src.shape[0], dst.shape[0])
        dst[:si, :sj] = src[:si, :sj]

        for i in range(1, r_i):
            i_min = i * src.shape[0]
            i_max = (i + 1) * src.shape[0]
            dst[i_min:i_max, :] = dst[:src.shape[0], :]

        if e_i != 0:
            dst[r_i * src.shape[0]:, :] = dst[:e_i, :]

        for j in range(1, r_j):
            j_min = j * src.shape[1]
            j_max = (j + 1) * src.shape[1]
            dst[:, j_min:j_max] = dst[:, :src.shape[1]]

        if e_j != 0:
            dst[:, r_j * src.shape[0]:] = dst[:, :e_j]

        return dst


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