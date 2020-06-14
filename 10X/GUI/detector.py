from tkinter import *
from tkinter import ttk, filedialog
from PIL import ImageTk, Image, ImageDraw, ImageFont
from models import Darknet
from utils.utils import *
import torchvision.transforms as transforms

cmap = plt.get_cmap("tab20b")
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class MY_GUI():
    def __init__(self, init_window_name, model=None):
        self.init_window_name = init_window_name
        self.conf_thres = 0.1
        self.nms_thres = 0.1
        # self.model = model
        self.x1 = 2000; self.y1 = 2000; self.x2 = 3920; self.y2 = 3216
        self.roi_w = 1920
        self.roi_h = 1216
        self.result_data_w = 960
        self.result_data_h = 608
        self.init_data_w = 500
        self.init_data_h = 500
        self.img = Image.open('test_10000.jpg')
        self.whole_img = self.img.copy()
        self.img_w = self.img.width
        self.img_h = self.img.height
        self.init_img = ImageTk.PhotoImage(self.whole_img.resize((self.init_data_w, self.init_data_h)))
        self.roi_img = self.img.crop([self.x1, self.y1, self.x2, self.y2])
        self.roi = ImageTk.PhotoImage(self.roi_img.resize((self.result_data_w, self.result_data_h)))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # def init_data(self):

    #设置窗口
    def set_init_window(self):

        self.init_window_name.title("detector_v0.1")
        #self.init_window_name.geometry('320x160+10+10')
        self.init_window_name.geometry('1500x731+10+10')
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
        self.init_data = Label(self.init_window_name, image=self.init_img) #原始数据录入框
        self.init_data.grid(row=1, column=0, rowspan=5, columnspan=5)
        self.result_data = Label(self.init_window_name, image=self.roi)  #处理结果展示
        self.result_data.grid(row=1, column=11, rowspan=10, columnspan=10)
        self.log_data_Text = Text(self.init_window_name, width=66, height=5)  # 日志框
        self.log_data_Text.grid(row=13, column=0, columnspan=10, rowspan=5)
        self.cmb_label1 = Label(self.init_window_name, text="选个检测模型")
        self.cmb_label1.grid(row=12, column=11)
        self.model_cmb = ttk.Combobox(self.init_window_name, width=60)
        self.model_cmb.grid(row=13, column=11, columnspan=5)
        self.model_cmb['value'] = ("config/yolov3-tiny-6b.cfg", "config/yolov3-tiny-4.cfg", "config/yolov3-tiny-5.cfg")
        self.model_cmb.current(0)
        self.cmb_label2 = Label(self.init_window_name, text="给模型选个权重")
        self.cmb_label2.grid(row=12, column=16)
        self.weight_path = StringVar(value="checkpoints_6b_nonsquare_randomAug/yolov3_ckpt_1024_4.pth")
        self.path_entry = Entry(self.init_window_name, width=70, textvariable=self.weight_path)
        self.path_entry.grid(row=13, column=16, columnspan=4)
        self.path_btn = Button(self.init_window_name, text="路径选择", command=self.selectPath)
        self.path_btn.grid(row=13, column=20)
        self.model_weights_btn = Button(self.init_window_name, text="set model", command=self.set_weights)
        self.model_weights_btn.grid(row=14, column=16)
        self.set_model()
        self.set_weights()
        #bind
        # self.init_window_name.bind('<KeyPress>', self.onclick)
        # self.str_trans_to_md5_button.bind('<Button-1>')
        self.init_data.bind("<Button-1>", self.click_roi)
        self.init_window_name.bind_all("<KeyPress-Up>", self.direction_buttons) #绑定方向键与函数
        self.init_window_name.bind_all("<KeyPress-Down>", self.direction_buttons)
        self.init_window_name.bind_all("<KeyPress-Left>", self.direction_buttons)
        self.init_window_name.bind_all("<KeyPress-Right>", self.direction_buttons)
        self.model_cmb.bind("<<ComboboxSelected>>", self.set_model)
        self.draw_init_rec()

    def selectPath(self):
        # 选择文件path_接收文件地址
        path_ = filedialog.askopenfilename()
        # 通过replace函数替换绝对文件地址中的/来使文件可被程序读取
        # 注意：\\转义后为\，所以\\\\转义后为\\
        path_ = path_.replace("/", "\\\\")
        # path设置path_的值
        self.weight_path.set(path_)
        self.set_weights()

    def set_weights(self):
        self.set_model()
        path = self.weight_path.get()
        try:
            if path.endswith(".pth"):
                self.model.load_state_dict(torch.load(path))
            else:
                self.model.load_darknet_weights(path)
            self.write_log_to_Text("Load weights SUCCESS")
            self.model.eval()
            self.refresh_results()
        except Exception as e:
            self.write_log_to_Text("Load weights ERROR:")
            self.write_log_to_Text(e)

    def set_model(self):
        try:
            self.model = Darknet(self.model_cmb.get()).to(self.device)
            self.write_log_to_Text("Load model_def SUCCESS")
        except Exception as e:
            self.write_log_to_Text("Load model_def ERROR:")
            self.write_log_to_Text(e)

    def draw_init_rec(self):
        draw = ImageDraw.Draw(self.whole_img)
        draw.rectangle([self.x1, self.y1, self.x2, self.y2], outline='red', width=30)
        self.back_img = ImageTk.PhotoImage(self.whole_img.resize((self.init_data_w, self.init_data_h)))
        self.init_data.config(image=self.back_img)

    def draw_detections(self, detections):

        draw = ImageDraw.Draw(self.roi_img)
        if detections is not None:
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                # print("\t+ Label: %s, Conf: %.5f" % ("pos", cls_conf.item()))

                # box_w = x2 - x1
                # box_h = y2 - y1
                txt = "%.2f" % conf
                # color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                draw.rectangle([x1, y1, x2, y2], outline='black', width=3)
                draw.text([x1, y1-30], txt, fill='black', font=ImageFont.truetype("arial", 30))
        self.roi = ImageTk.PhotoImage(self.roi_img.resize((self.result_data_w, self.result_data_h)))
        self.result_data.config(image=self.roi)

    def click_roi(self, event):
        nx = event.x*(self.img_w/self.init_data_w)
        ny = event.y*(self.img_h/self.init_data_h)
        self.x1 = int(nx - (self.roi_w/2)); self.x2 = int(nx + (self.roi_w/2))
        self.y1 = int(ny - (self.roi_h/2)); self.y2 = int(ny + (self.roi_h/2))
        self.refresh_results()

    def direction_buttons(self, event):# 绑定方向键
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
        self.refresh_results()

    def refresh_results(self):
        print(self.x1, self.y1, self.x2, self.y2)
        self.whole_img = self.img.copy()
        self.roi_img = self.img.crop([self.x1, self.y1, self.x2, self.y2])
        detections = self.model(get_input(self.roi_img))
        detections = non_max_suppression(detections, nms_thres=self.nms_thres)
        # detections = [[500, 500, 800, 800, 1, 1, 1]]
        self.draw_detections(detections[0])
        self.draw_init_rec()
        if detections[0] is not None:
            self.det_num = str(detections[0].shape[0])
        else:
            self.det_num = None
        logmsg = 'Detection in (%d, %d): %s个阳性细胞' % (self.x1 + (self.roi_w/2), self.y1 + (self.roi_h/2), self.det_num)
        self.write_log_to_Text(logmsg)

    #获取当前时间
    def get_current_time(self):
        current_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        return current_time

    #日志动态打印
    def write_log_to_Text(self,logmsg):
        current_time = self.get_current_time()
        logmsg_in = str(current_time) +" " + str(logmsg) + "\n"      #换行
        self.log_data_Text.insert(0.0, logmsg_in)

def get_input(imgs):
    imgs = transforms.ToTensor()(imgs).unsqueeze(0)
    return Variable(imgs.type(Tensor))

def draw_rectangle(img, obj, coord, w, h):
    x1, y1, x2, y2 = coord
    draw0 = ImageDraw.Draw(img)
    draw0.rectangle([x1, y1, x2, y2], outline='red', width=30)
    img_tk = ImageTk.PhotoImage(img.resize((w, h)))
    obj.config(image=img_tk)

def gui_start():
    init_window = Tk()
    # model =
    ZMJ_PORTAL = MY_GUI(init_window)
    # 设置根窗口默认属性
    ZMJ_PORTAL.set_init_window()

    init_window.mainloop()

gui_start()