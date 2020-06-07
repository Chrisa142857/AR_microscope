from tkinter import *
import time
from PIL import ImageTk, Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

LOG_LINE_NUM = 0
cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in np.linspace(0, 1, 20)]

class MY_GUI():
    def __init__(self, init_window_name):
        self.init_window_name = init_window_name
        self.conf_thres = 0.0001
        self.nms_thres = 0.1
        # self.model = model
        self.x1 = 2000; self.y1 = 2000; self.x2 = 3920; self.y2 = 3216

    def onclick(self, event):
        print(event.keycode)

    #设置窗口
    def set_init_window(self):
        self.init_window_name.title("detector_v0.1")
        #self.init_window_name.geometry('320x160+10+10')
        self.init_window_name.geometry('1473x731+10+10')
        #self.init_window_name["bg"] = "pink"                                    #窗口背景色，其他背景色见：blog.csdn.net/chl0000/article/details/7657887
        #self.init_window_name.attributes("-alpha",0.9)                          #虚化，值越小虚化程度越高
        #标签
        self.init_data_label = Label(self.init_window_name, text="待处理数据")
        self.init_data_label.grid(row=0, column=0)
        self.result_data_label = Label(self.init_window_name, text="输出结果")
        self.result_data_label.grid(row=0, column=11)
        self.log_label = Label(self.init_window_name, text="日志")
        self.log_label.grid(row=12, column=0)
        #文本框
        self.img = Image.open('test_10000.jpg')
        self.whole_img = self.img.copy()
        self.init_img = ImageTk.PhotoImage(self.whole_img.resize((500, 500)))
        self.roi_img = self.img.crop([self.x1, self.y1, self.x2, self.y2])
        self.roi = ImageTk.PhotoImage(self.roi_img.resize((960, 608)))
        self.init_data = Label(self.init_window_name, image=self.init_img) #原始数据录入框
        self.init_data.grid(row=1, column=0, rowspan=10, columnspan=10)
        self.result_data = Label(self.init_window_name, image=self.roi)  #处理结果展示
        self.result_data.grid(row=1, column=11, rowspan=10, columnspan=10)
        self.log_data_Text = Text(self.init_window_name, width=66, height=5)  # 日志框
        self.log_data_Text.grid(row=14, column=0, columnspan=10)
        # self.init_window_name.bind('<KeyPress>', self.onclick)
        # self.str_trans_to_md5_button.bind('<Button-1>')
        self.init_window_name.bind_all("<KeyPress-Up>", self.refresh) #绑定方向键与函数
        self.init_window_name.bind_all("<KeyPress-Down>", self.refresh)
        self.init_window_name.bind_all("<KeyPress-Left>", self.refresh)
        self.init_window_name.bind_all("<KeyPress-Right>", self.refresh)
        draw0 = ImageDraw.Draw(self.whole_img)
        draw0.rectangle([self.x1, self.y1, self.x2, self.y2], outline='red', width=30)
        self.back_img = ImageTk.PhotoImage(self.whole_img.resize((500, 500)))
        self.init_data.config(image=self.back_img)

    def refresh(self, event):  # 绑定方向键
        self.whole_img = self.img.copy()
        dx = 0; dy = 0
        if event.keysym == "Up":
            dy = -100
        elif event.keysym == "Down":
            dy = 100
        elif event.keysym == "Left":
            dx = -100
        elif event.keysym == "Right":
            dx = 100
        self.x1 += dx; self.x2 += dx; self.y1 += dy; self.y2 += dy
        self.roi_img = self.img.crop([self.x1, self.y1, self.x2, self.y2])
        # detections = self.model(self.roi_img)
        # # detections = non_max_suppression(detections, self.conf_thres, self.nms_thres)
        # unique_labels = detections[:, -1].cpu().unique()
        # bbox_colors = colors[0]
        #
        # for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        #     print("\t+ Label: %s, Conf: %.5f" % ("pos", cls_conf.item()))
        #
        #     box_w = x2 - x1
        #     box_h = y2 - y1
        #
        #     color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]

        draw = ImageDraw.Draw(self.roi_img)
        draw.rectangle([500, 500, 800, 800], outline='black', width=3)
        self.roi = ImageTk.PhotoImage(self.roi_img.resize((960, 608)))
        self.result_data.config(image=self.roi)
        draw0 = ImageDraw.Draw(self.whole_img)
        draw0.rectangle([self.x1, self.y1, self.x2, self.y2], outline='red', width=30)
        self.back_img = ImageTk.PhotoImage(self.whole_img.resize((500, 500)))
        self.init_data.config(image=self.back_img)
        logmsg = 'Detection in (%d, %d): ' % (self.x1, self.y1)
        self.write_log_to_Text(logmsg)

    #获取当前时间
    def get_current_time(self):
        current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        return current_time


    #日志动态打印
    def write_log_to_Text(self,logmsg):
        global LOG_LINE_NUM
        current_time = self.get_current_time()
        logmsg_in = str(current_time) +" " + str(logmsg) + "\n"      #换行
        if LOG_LINE_NUM <= 7:
            self.log_data_Text.insert(END, logmsg_in)
            LOG_LINE_NUM = LOG_LINE_NUM + 1
        else:
            self.log_data_Text.delete(1.0,2.0)
            self.log_data_Text.insert(END, logmsg_in)


def gui_start():
    init_window = Tk()
    ZMJ_PORTAL = MY_GUI(init_window)
    # 设置根窗口默认属性
    ZMJ_PORTAL.set_init_window()

    init_window.mainloop()

gui_start()