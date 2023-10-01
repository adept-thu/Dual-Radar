#! /usr/bin/python3
import os

if __name__ == '__main__' :
    os.system("ls")
    cur_file  = os.path.abspath(__file__)
    cur_dir = cur_file.rsplit("/", 1)[0]
    os.chdir(f"OpenPCDet/tools") # 改变到当前文件所在目录（OpenPCDet/tools） 
    # 为了接下来执行bash_train.sh 不然会找不到bash_train.sh文件
    os.system("bash bash_train.sh") # 执行sh脚本