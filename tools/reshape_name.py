import os
# path = r"D:\PyCharm\Project\pp\zhongqi1\training\0136\training\image_2"
# path0 = r"D:\PyCharm\Project\pp\zhongqi1\training\0136\training\velodyne"
# path1 = r"D:\PyCharm\Project\pp\zhongqi1\training\0136\training\arbe"
# path2 = r"D:\PyCharm\Project\pp\zhongqi1\training\0136\training\ars548"
# path3 = r"D:\PyCharm\Project\pp\zhongqi1\training\0136\training\json"

#改变
path = "/home/fangcheng/dataset/0009/testing/image_2/"
path0 = "/home/fangcheng/dataset/0009/testing/ars548/"
path1 = "/home/fangcheng/dataset/0009/testing/arbe/"
path2 = "/home/fangcheng/dataset/0009/testing/label_2/"
path3 = "/home/fangcheng/dataset/0009/testing/calib/"



filelist = os.listdir(path)
filelist.sort()
i = 0
for file in filelist:
    Olddir=os.path.join(path,file)
    if os.path.isdir(Olddir):
        continue
    filename=os.path.splitext(file)[0]  #分离文件名与扩展名；默认返回(fname,fextension)元组
    # filename=filename[3:]  #忽略文件名前3位 取后面数字字符串
    filetype = os.path.splitext(file)[1]  #文件后缀名 例如.jpg
    Newdir=os.path.join(path,str(i).zfill(6)+filetype)  #xxx可改为任意前缀，后面是6位整数
    os.rename(Olddir,Newdir)
    i = i + 1

filelist = os.listdir(path0)
filelist.sort()
i = 0
for file in filelist:
    Olddir=os.path.join(path0,file)
    if os.path.isdir(Olddir):
        continue
    filename=os.path.splitext(file)[0]  #分离文件名与扩展名；默认返回(fname,fextension)元组
    # filename=filename[3:]  #忽略文件名前3位 取后面数字字符串
    filetype = os.path.splitext(file)[1]  #文件后缀名 例如.jpg
    Newdir=os.path.join(path0,str(i).zfill(6)+filetype)  #xxx可改为任意前缀，后面是6位整数
    os.rename(Olddir,Newdir)
    i = i + 1

filelist = os.listdir(path1)
filelist.sort()
i = 0
for file in filelist:
    Olddir=os.path.join(path1,file)
    if os.path.isdir(Olddir):
        continue
    filename=os.path.splitext(file)[0]  #分离文件名与扩展名；默认返回(fname,fextension)元组
    # filename=filename[3:]  #忽略文件名前3位 取后面数字字符串
    filetype = os.path.splitext(file)[1]  #文件后缀名 例如.jpg
    Newdir=os.path.join(path1,str(i).zfill(6)+filetype)  #xxx可改为任意前缀，后面是6位整数
    os.rename(Olddir,Newdir)
    i = i + 1

filelist = os.listdir(path2)
filelist.sort()
i = 0
for file in filelist:
    Olddir=os.path.join(path2,file)
    if os.path.isdir(Olddir):
        continue
    filename=os.path.splitext(file)[0]  #分离文件名与扩展名；默认返回(fname,fextension)元组
    # filename=filename[3:]  #忽略文件名前3位 取后面数字字符串
    filetype = os.path.splitext(file)[1]  #文件后缀名 例如.jpg
    Newdir=os.path.join(path2,str(i).zfill(6)+filetype)  #xxx可改为任意前缀，后面是6位整数
    os.rename(Olddir,Newdir)
    i = i + 1

filelist = os.listdir(path3)
filelist.sort()
i = 0
for file in filelist:
    Olddir=os.path.join(path3,file)
    if os.path.isdir(Olddir):
        continue
    filename=os.path.splitext(file)[0]  #分离文件名与扩展名；默认返回(fname,fextension)元组
    # filename=filename[3:]  #忽略文件名前3位 取后面数字字符串
    filetype = os.path.splitext(file)[1]  #文件后缀名 例如.jpg
    Newdir=os.path.join(path3,str(i).zfill(6)+filetype)  #xxx可改为任意前缀，后面是6位整数
    os.rename(Olddir,Newdir)
    i = i + 1

