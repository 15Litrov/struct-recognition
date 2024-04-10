import imgui
import numpy as np
import pickle
import glfw
import scipy.stats as stats

class Container:
    def __init__(self):
        pass

class Lab10Host:
    def __init__(self):
        self.openedTasks = set()
        self.password = "qwerty"
        self.password_input = ""
        self.time = glfw.get_time()
        self.carret_pos = 0
        self.history = []
        self.times = np.empty(len(self.password))
        self.input_id = 0
        self.significance_level = 0.01
        self.desired_prop = 0.85

        self.auth_history = []
        self.auth_id = 0
        self.auth_tries = 2
        self.auth_times = np.empty(len(self.password))
        self.auth_state = ["In process"]

        self.test_times = np.empty(len(self.password))
        self.test_id = 0
        self.test_num = 0
        self.carret_pos = 0
        self.password_input = ""
        self.test_count = 0
        self.test_N = 0
        self.test_authorized = True

    def update(self):
        if imgui.begin_main_menu_bar().opened:
            if imgui.menu_item("Train")[0]:
                self.openedTasks.add(self.task_Train)

            if imgui.menu_item("Test")[0]:
                self.openedTasks.add(self.task_Test)

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

    def callback_train_input(self, data):
        if data.event_char != self.password[self.carret_pos]:
            return 1
        
        t = glfw.get_time()
        dt = t - self.time
        self.time = t

        self.times[self.carret_pos] = dt
        self.carret_pos += 1

        if self.carret_pos == len(self.password):
            cleaned = self.remove_mistakes(self.times[1:])
            if np.any(cleaned > 0):
                self.history.append(cleaned)

            self.times = np.empty(len(self.password))
            self.carret_pos = 0
            self.password_input = ""
            self.input_id += 1
            return 1

        return 0

    def task_Auth(self):
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        with imgui.begin("Authentication", True, flags) as window:
            if not window.opened:
                self.openedTasks.remove(self.task_Auth)
                self.auth_times = np.empty(len(self.password))
                self.auth_history = []
                self.auth_id = 0
                self.carret_pos = 0
                self.password_input = ""
                self.auth_state = ["In process"]
                return
            
            imgui.text_ansi(f"Authentication try: {len(self.auth_history) + 1}/{self.auth_tries}")

            if self.carret_pos != 0:
                imgui.text_ansi(self.password[:self.carret_pos])
                imgui.same_line()

            imgui.text_ansi_colored(self.password[self.carret_pos], 0, 1, 0)

            if self.carret_pos != len(self.password) - 1:
                imgui.same_line()
                imgui.text_ansi(self.password[self.carret_pos + 1:])

            _, _ = imgui.input_text(f"Password##auth{self.auth_id}", self.password_input, -1, imgui.INPUT_TEXT_PASSWORD | imgui.INPUT_TEXT_CALLBACK_CHAR_FILTER, self.callback_auth_input)

            for state in reversed(self.auth_state):
                imgui.text_ansi(state)
    


    def callback_auth_input(self, data):
        if data.event_char != self.password[self.carret_pos]:
            return 1
        
        t = glfw.get_time()
        dt = t - self.time
        self.time = t

        self.auth_times[self.carret_pos] = dt
        self.carret_pos += 1

        if self.carret_pos == len(self.password):
            self.carret_pos = 0
            self.password_input = ""

            cleaned = self.remove_mistakes(self.auth_times[1:])
            if np.any(cleaned > 0):
                self.auth_history.append(cleaned)

            self.auth_id += 1
            if len(self.auth_history) == self.auth_tries:
                success, self.auth_state[-1], _, _ = self.authenticate(self.auth_history)
                if success:
                    for h in self.auth_history:
                        self.history.append(h)

                self.auth_state.append("In progress")
                self.auth_history = []

            self.auth_times = np.empty(len(self.password))


            return 1

        return 0

    def task_Test(self):
        flags = imgui.WINDOW_ALWAYS_AUTO_RESIZE
        with imgui.begin("Test", True, flags) as window:
            if not window.opened:
                self.openedTasks.remove(self.task_Test)
                self.test_times = np.empty(len(self.password))
                self.test_id = 0
                self.test_num = 0
                self.carret_pos = 0
                self.password_input = ""
                self.test_count = 0
                self.test_N = 0
                return
            
            changed, self.test_authorized = imgui.checkbox("Authorized", self.test_authorized)
            if changed:
                self.test_num = 0
                self.carret_pos = 0
                self.password_input = ""
                self.test_times = np.empty(len(self.password))
                self.test_count = 0
                self.test_N = 0

            imgui.text_ansi(f"Test try: {self.test_num + 1}")

            if self.carret_pos != 0:
                imgui.text_ansi(self.password[:self.carret_pos])
                imgui.same_line()

            imgui.text_ansi_colored(self.password[self.carret_pos], 0, 1, 0)

            if self.carret_pos != len(self.password) - 1:
                imgui.same_line()
                imgui.text_ansi(self.password[self.carret_pos + 1:])

            _, _ = imgui.input_text(f"Password##test{self.test_id}", self.password_input, -1, imgui.INPUT_TEXT_PASSWORD | imgui.INPUT_TEXT_CALLBACK_CHAR_FILTER, self.callback_test_input)

            if self.test_N > 0:
                if self.test_authorized:
                    imgui.text_ansi(f"1st kind error: {1 - self.test_count / float(self.test_N)}")
                else:
                    imgui.text_ansi(f"2nd kind error: {self.test_count / float(self.test_N)}")

    def callback_test_input(self, data):
        if data.event_char != self.password[self.carret_pos]:
            return 1
        
        t = glfw.get_time()
        dt = t - self.time
        self.time = t

        self.test_times[self.carret_pos] = dt
        self.carret_pos += 1

        if self.carret_pos == len(self.password):
            self.carret_pos = 0
            self.password_input = ""

            cleaned = self.remove_mistakes(self.test_times[1:])
            if np.any(cleaned > 0):
                _, _, count, N = self.authenticate([cleaned])
                self.test_count += count
                self.test_N += N

            self.test_id += 1
            self.test_num += 1
            self.test_times = np.empty(len(self.password))


            return 1

        return 0

    def authenticate(self, auth_history):
        n = len(self.password) - 1

        Sx = []
        nx = []

        for h in self.history:
            indices = np.where(h > 0)
            val = h[indices]
            nx.append(val.shape[0])
            Sx.append(np.var(val))

        Sy = []
        ny = []

        for h in auth_history:
            indices = np.where(h > 0)
            val = h[indices]
            ny.append(val.shape[0])
            Sy.append(np.var(val))

        Sx = np.array(Sx)
        Sy = np.array(Sy)

        i_Sxmax = np.argmax(Sx)
        i_Symax = np.argmax(Sy)

        Smax, Smin, nmax, nmin = 0, 0, 0, 0
        if (Sx[i_Sxmax] > Sy[i_Symax]):
            Smax = Sx[i_Sxmax]
            nmax = nx[i_Sxmax] - 1
            Smin = Sy[i_Symax]
            nmin = ny[i_Symax] - 1
        else:
            Smin = Sx[i_Sxmax]
            nmin = nx[i_Sxmax] - 1
            Smax = Sy[i_Symax]
            nmax = ny[i_Symax] - 1

        Fp = Smax / Smin
        Ft = stats.f.ppf(1 - self.significance_level, nmax, nmin)
        if Fp > Ft:
            return False, f"Rejected ({Fp}/{Ft})", 0, len(self.history) * len(auth_history)
        
        Sx, Mx, Sy, My = [], [], [], []
        for h in self.history:
            indices = np.where(h > 0)
            v = h[indices]
            Mx.append(np.mean(v))
            Sx.append(np.var(v))

        for h in auth_history:
            indices = np.where(h > 0)
            v = h[indices]
            My.append(np.mean(v))
            Sy.append(np.var(v))

        count = 0
        alpha = 1 - self.desired_prop
        Tt = stats.t.ppf(1 - alpha/2, n - 1)
        for i in range(len(Mx)):
            Sxi = Sx[i]
            Mxi = Mx[i]
            for j in range(len(My)):
                Syi = Sy[j]
                Myi = My[j]

                S = np.sqrt((Sxi + Syi)*(n - 1) / (2*n -1))
                Tp = np.abs(Mxi - Myi) / (S * np.sqrt(2 / n))

                if Tp < Tt:
                    count += 1

        N = (len(Mx) * len(My))
        P = count / (len(Mx) * len(My))
        if P >= self.desired_prop:
            return True, f"Accepted {P}", count, N
        else:
            return False, f"Rejected {P}", count, N


    def remove_mistakes(self, series):
        n = len(self.password) - 1
        significant = [-1]*series.shape[0]

        M = [None]*series.shape[0]
        for i in range(series.shape[0]):
            Mi = 0
            for j in range(series.shape[0]):
                if i == j:
                    continue

                Mi += series[j]

            M[i] = Mi / (n - 1)

        S = [None]*n
        for i in range(n):
            Si = 0
            Mi = M[i]
            for j in range(n):
                if i == j:
                    continue

                Si += (series[j] - Mi)**2

            S[i] = np.sqrt(Si / (n - 2))

        Tt = stats.t.ppf(1 - self.significance_level/2, n - 2)
        for i in range(n):
            Tpi = np.abs(series[i] - M[i]) / S[i]
            if Tpi < Tt:
                significant[i] = series[i]

        return np.array(significant)