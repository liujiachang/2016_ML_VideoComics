import tornado.web
import os
import glob
from start import *

class UploadHandler(tornado.web.RequestHandler):  #上传文件
    def get(self,*args,**kwargs):
        self.render('web.html')

    def post(self,*args,**kwargs):
        files = self.request.files.get('file',None)  #获取上传文件数据，返回文件列表

        for file in files: #可能同一个上传的文件会有多个文件，所以要用for循环去迭代它
            # filename 文件的实际名字，body 文件的数据实体；content_type 文件的类型。 这三个对象属性可以像字典一样支持关键字索引
            save_to = './datasets/mp4/{}'.format(file['filename'])
            #以二进制格式打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，即原有内容会被删除。如果该文件不存在，创建新文件。一般用于非文本文件如图片等。
            with open(save_to,'wb') as f: #二进制
                f.write(file['body'])
            start()
        self.redirect('/result')

class ResultHandler(tornado.web.RequestHandler):
    """
     结果展示页面
    """
    def get(self,*args,**kwargs):
        # image_urls = get_images("./static/uploads")  #打开指定路径下的文件，或者static/uploads
        os.chdir('static')  # 用于改变当前工作目录到指定的路径
        img_source = glob.glob("./source"+'/*.jpg')
        img_results = glob.glob("./results"+'/*.jpg')
        os.chdir("..")
        self.render('result.html',img_source=img_source,img_results=img_results)










































# from tornado.ioloop import IOLoop
# from tornado import web
# import shutil
# import glob
# import os
# import json
#
#
# class FileUploadHandler(web.RequestHandler):
#     def get(self):
#         os.chdir("./frontend")
#         logo = glob.glob("./img/logo.png")
#         os.chdir("..")
#         self.render("./frontend/web.html",logo = logo)
#
#     def post(self):
#         os.chdir("./datasets")
#         source = glob.glob("source" + "/*.jpg")
#         result = glob.glob("result" + "/*.jpg")
#         os.chdir("..")
#
#         file_metas = self.request.files.get('file', None)  # 提取表单中‘name’为‘file’的文件元数据
#         if file_metas:
#             fin = open("./datasets/mp4/01.mp4", "w")
#             print("success to open file")
#             fin.write(file_metas)
#             fin.close()
#
#         self.render("./frontend/result.html",source = source,result = result)
#
#
#
# application = web.Application([(r'/file', FileUploadHandler)],autoreload = True)
# #开启服务器监听
# application.listen(7000)
# IOLoop.current().start()
#

