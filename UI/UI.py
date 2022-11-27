###################################### 主UI #######################################

from tkinter import *
from tkinter.messagebox import *
from tkinter import ttk
from PIL import Image, ImageTk
import cv2


# 設計登入頁面
class LoginPage(object):
    def __init__(self, master=None):
        self.root = master  # 定義內部變數root
        self.root.geometry('%dx%d' % (300, 300))  # 設定視窗大小
        self.username = StringVar()
        self.username.set('admin')
        self.password = StringVar('')
        self.password.set('123456')
        self.createPage()

    def createPage(self):
        self.page = Frame(self.root)  # 建立Frame
        self.page.pack()
        Label(self.page, text='Code Reader 登入系統').pack()
        Label(self.page).pack()
        Label(self.page, text='賬戶: ').pack()
        Entry(self.page, textvariable=self.username).pack()
        Label(self.page, text='密碼: ').pack()
        Entry(self.page, textvariable=self.password, show='*').pack()
        Button(self.page, text='登入', command=self.loginCheck).pack()
        Button(self.page, text='退出', command=self.page.quit).pack()

    def loginCheck(self):
        name = self.username.get()
        secret = self.password.get()
        if name == 'admin' and secret == '123456':
            self.page.destroy()
            MainPage(self.root)
        else:
            showinfo(title='錯誤', message='賬號或密碼錯誤！')

        # 設計主程式頁面


class MainPage(object):
    def __init__(self, master=None):
        self.root = master  # 定義內部變數root
        self.root.geometry('%dx%d' % (300, 350))  # 設定視窗大小
        self.createPage()

    def createPage(self):
        self.inputPage = InputFrame(self.root)  # 建立不同Frame
        self.recordPage = RecordFrame(self.root)
        self.resultPage = ResultFrame(self.root)
        self.inputPage.pack()  # 預設顯示資料錄入介面
        menubar = Menu(self.root)
        menubar.add_command(label='功能選擇', command=self.inputData)
        menubar.add_command(label='紀錄查詢', command=self.recordDisp)
        menubar.add_command(label='辨識結果', command=self.resultDisp)
        self.root['menu'] = menubar  # 設定選單欄

    def inputData(self):
        self.recordPage.pack_forget()
        self.resultPage.pack_forget()
        self.inputPage = InputFrame(self.root)  # 建立不同Frame
        self.inputPage.pack()

    def recordDisp(self):
        self.inputPage.pack_forget()
        self.resultPage.pack_forget()
        self.recordPage = RecordFrame(self.root)
        self.recordPage.pack()

    def resultDisp(self):
        self.inputPage.pack_forget()
        self.recordPage.pack_forget()
        self.resultPage = ResultFrame(self.root, root.key_value_dict, root.exe_time, root.combined_result)
        self.resultPage.pack()

    # 設計功能選擇頁面


class InputFrame(Frame):  # 繼承Frame類
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定義內部變數root
        self.folder_name = StringVar()                          
        self.folder_name.set('')
        self.box = ttk.Combobox(root, textvariable=self.folder_name, state='readonly', values=['siliconlab_box_1','siliconlab_box_2','siliconlab_box_3','skywork_box','skywork_disk','STM_disk','melexis_disk','multi_code'])
        self.createPage()

    def createPage(self):
        Label(self, text='Code Reader 功能選擇').pack()
        Label(self).pack()
        Label(self, text='即時錄影偵測: ').pack()
        Button(self, text='開始偵測', command=self.real_time_obj_detection).pack()
        Label(self, text='本地相片偵測: ').pack()
        Label(self, text='本地相片偵測: ').pack()
        Button(self, text='開始偵測', command=self.photo_obj_detection).pack()
        Label(self, text='雲端相片偵測: ').pack()
        Button(self, text='開始偵測', command=self.photo_obj_detection_cloud).pack()
        Label(self, text='跨面標籤偵測: ').pack()
        Button(self, text='開始偵測', command=self.cross_photo_obj_detection).pack()
        
        # 設置comboBox讓使用者選擇要用哪個資料夾
        self.box.pack()
        
    def real_time_obj_detection(self):
        print('real_time_obj_detection')
        self.folder_name.set(f'{self.box.current()}:{self.box.get()}')
        folder_name = self.folder_name.get().split(':')[1]
        root.destroy()
#         real_time_obj_detection(model_path,GPU_ratio=GPU_ratio,toCSV=True,folder_name)


    def photo_obj_detection(self):
        print('photo_obj_detection')
        self.folder_name.set(f'{self.box.current()}:{self.box.get()}')
        folder_name = self.folder_name.get().split(':')[1]
        root.destroy()
#         photo_obj_detection(model_path, GPU_ratio=GPU_ratio, toCSV=True,folder_name)

    def photo_obj_detection_cloud(self):
        print('photo_obj_detection_cloud')
        self.folder_name.set(f'{self.box.current()}:{self.box.get()}')
        folder_name = self.folder_name.get().split(':')[1]
        root.destroy()
#         photo_obj_detection_cloud(model_path, GPU_ratio=GPU_ratio, toCSV=True,folder_name)

    def cross_photo_obj_detection(self):
        print('cross_photo_obj_detection')
        self.folder_name.set(f'{self.box.current()}:{self.box.get()}')
        folder_name = self.folder_name.get().split(':')[1]
        root.destroy()
#         cross_photo_obj_detection(model_path, GPU_ratio=GPU_ratio, toCSV=True,folder_name)


class RecordFrame(Frame):  # 繼承Frame類
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定義內部變數root
        self.itemName = StringVar()
        self.createPage()

    def createPage(self):
        Label(self, text='查詢介面').pack()


class ResultFrame(Frame):  # 繼承Frame類
    def __init__(self, master=None, key_value_dict=[], exe_time=0, combined_result=[]):
        Frame.__init__(self, master)
        self.key_value_dict = key_value_dict
        self.exe_time = exe_time
        self.combined_result = combined_result
        self.root = master  # 定義內部變數root
        self.createPage()

    def inputData(self):
        self.pack_forget()

    def createPage(self):
        # 設定變數
        col_name_list = ['PN', 'DATE', 'QTY', 'LOT', 'COO']
        key_value_list = []

        # init的時候甚麼都不做
        if self.key_value_dict == []:
            # 顯示頁面標題: "尚無辨識結果"
            label = Label(self, text="尚無辨識結果", font=("Arial", 20, "bold"), padx=5, pady=5, fg="black").pack()
        else:
            # 轉換輸入資訊
            for col in col_name_list:
                now_label_id = 0
                col_name_value_list = []
                exist_flag = False
                for diction in self.key_value_dict:
                    if diction['label_id'] != now_label_id:
                        now_label_id = now_label_id + 1
                        if exist_flag == False:
                            col_name_value_list.append('')
                    for key in diction.keys():
                        if key == col:
                            exist_flag = True
                            col_name_value_list.append(diction.get(key))
                if len(col_name_value_list) != now_label_id + 1:
                    col_name_value_list.append('')
                key_value_list.append(col_name_value_list)
            label_data_list = []

            for i in range(len(key_value_list[0])):
                data_list = []
                for col_value_list in key_value_list:
                    data_list.append(col_value_list[i])
                label_data_list.append(data_list)

            # 如果要印出decode結果，則加長UI
            #             if self.combined_result:
            #                 height = 650
            #             else:
            #                 height = 350

            # 顯示頁面標題: "Code Reader"
            label = Label(text="Code Reader", font=("Arial", 20, "bold"), padx=5, pady=5, fg="black")
            label.pack()

            # 顯示當前圖片
            img_open = Image.open(r'普通標籤.jpg')
            img_open_width, img_open_height = img_open.size
            # 自動調整圖片大小
            resize_factor = 300
            if img_open_width / img_open_height >= 1:
                img_open = img_open.resize((int(img_open_width / img_open_height * resize_factor), resize_factor))

            else:
                img_open = img_open.resize((resize_factor, int(img_open_height / img_open_width * resize_factor)))
            img_png = ImageTk.PhotoImage(img_open)
            label_img = Label(bg='gray94', fg='blue', padx=5, pady=25, image=img_png).pack()

            # 加入辨識結果對應表格
            tree = ttk.Treeview(root, height=len(label_data_list), padding=(10, 5, 20, 20),
                                columns=('PN', 'Date', 'QTY', 'LOT', 'COO'))
            tree.column("PN", width=200)
            tree.column("Date", width=100)
            tree.column("QTY", width=100)
            tree.column("LOT", width=200)
            tree.column("COO", width=100)

            tree.heading("PN", text="PN")
            tree.heading("Date", text="Date")
            tree.heading("QTY", text="QTY")
            tree.heading("LOT", text="LOT")
            tree.heading("COO", text="COO")

            for i, data_list in enumerate(label_data_list):
                tree.insert("", i, text=i, values=data_list)  # 插入資料，
            tree.pack()

        

            # 顯示辨識時間
            label = Label(text=f"執行時間: {self.exe_time:.2} (s)", font=("Arial", 14, "bold"), padx=5, pady=25, fg="black")
            label.pack()
            button = Button(text='繼續').pack()
            button = Button(text='回到主頁面', command=self.inputData).pack()
            root.mainloop()  # 執行視窗
