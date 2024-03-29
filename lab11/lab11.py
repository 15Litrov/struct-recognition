import imgui
import numpy as np
import pickle

class Lab11Host:
    def __init__(self):
        self.openedTasks = set()
        self.dict_path = "E:\STUDY\struct-recognition\lab11\cities_names_dict.pkl"
        self.dict = None
        self.threshold = 600
        self.text_to_score = ""
        self.score_text_text = ""
        self.append_endl = False

    def update(self):
        if imgui.begin_main_menu_bar().opened:
            if imgui.menu_item("Plausibility")[0]:
                self.openedTasks.add(self.task_Score)

            imgui.end_main_menu_bar()

        for task in list(self.openedTasks):
                task()

    def task_Score(self):
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        with imgui.begin("Plausibility", True, flags) as window:
            if not window.opened:
                self.openedTasks.remove(self.task_Score)
                self.inited = False
                return
            
            changed1, text_val_1 = imgui.input_text("Dictionary path:", self.dict_path)
            if changed1:
                self.dict_path = text_val_1

            if imgui.button("Load dictionary"):
                try:
                    with open(self.dict_path, 'rb') as f:
                        self.dict = pickle.load(f)
                except:
                    self.dict = None

            _, self.threshold = imgui.input_int("Threshold:", self.threshold)

            imgui.separator()

            _, self.append_endl = imgui.checkbox("Append '\\n'", self.append_endl)
            _, self.text_to_score = imgui.input_text("Text:", self.text_to_score)


            if imgui.button("Score"):
                if self.dict is None:
                    self.score_text_text = "Please, load the dictionary"

                score = int(self.score_text(self.text_to_score, self.dict, self.append_endl))
                self.score_text_text = f"Plausibility: {score}. "
                if score > self.threshold:
                    self.score_text_text += f"Realistic ({score} > {self.threshold})"
                else:
                    self.score_text_text += f"Fake ({score} <= {self.threshold})"

            imgui.separator()
            imgui.text_ansi(self.score_text_text)


    def score_text(self, text, dict, append_endl):
        if append_endl:
            text += "\n"

        l = len(text) - 1
        p = 1.0 / l
        score = 1
        for i in range(l):
            couple = text[i:i+2]
            if couple not in dict.keys():
                continue

            score *= np.power(dict[couple], p)

        return score
