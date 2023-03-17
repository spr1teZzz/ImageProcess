#生成100万个区间值直觉模糊数，
# 要求：
# （1）没有重复的，
# （2）u-,u+,v-,v+保留小数点3、4、5、6位,即小数点3位到6位不能为0.
#  (3)u-<u+,v-<v+, (u+)+(v+)<=1
# （4）保存在CSV文件中

import random  
import csv
from threading import Timer
import time
import schedule

SIZE = 10

flag = False
def decimal_digits(val: float):
    val_str = str(val)
    digits_location = val_str.find('.')
    if digits_location:
        return len(val_str[digits_location + 1:])

def judge(count):
    print('judge-----------')
    global flag
    if count[0]+count[1]+count[2]+count[3]+count[4] > SIZE:
        print('quit')
        flag = True
       
set1 = set()#存储0.0-0.2
set2 = set()#存储0.2-0.4
set3 = set()#存储0.4-0.6
set4 = set()#存储0.6-0.8
set5 = set()#存储0.8-1.0
# schedule.every().second.do(judge,[len(set1),len(set2),len(set3),len(set4),len(set5)])    # 每隔十分钟执行一次任务
time_start = time.time()  # 记录开始时间
while True:
    num = random.randint(3,6)#随机生成3-6位小数
    u1 = round(random.random(),num)
    u2 = round(random.random(),num)
    v1 = round(random.random(),num)
    v2 = round(random.random(),num)
    digit = decimal_digits(u1) + decimal_digits(u2) + decimal_digits(v1) + decimal_digits(v2)
    if  digit<= 24 and digit>=12 and v2<v1 and u2<u1 and u1+u2<=1:
        #满足开始插入set
        # tmpStr = str(u1)[2:]+ str(u2)[2:]+ str(v1)[2:]+ str(v2)[2:]
        
        #将u、v补成指定长度6位，便于后续分割
        u1 = '%.6f'%u1
        u2 = '%.6f'%u2
        v1 = '%.6f'%v1
        v2 = '%.6f'%v2

        # tmpStr = str(u1)+ str(u2)+ str(v1)+ str(v2)
        tmpStr = u1+ u2+ v1+ v2
        print('tmpStr:'+tmpStr)
        if float(u1)<=0.2:
            set1.add(tmpStr)
            pass
        elif float(u1)<=0.4:
            set2.add(tmpStr)
            pass
        elif float(u1)<=0.6:
            set3.add(tmpStr)
            pass
        elif float(u1)<=0.8:
            set4.add(tmpStr)
            pass
        else :
            set5.add(tmpStr)
            pass
    # schedule.every(10).minutes.do(judge(len(set1),len(set2),len(set3),len(set4),len(set5)))    # 每隔十分钟执行一次任务
    # schedule.run_pending() # 运行所有可运行的任务
    if len(set1)+len(set2)+len(set3)+len(set4)+len(set5)>=SIZE:
        break

#写入
header = ['U+','U-','V+','V-']
file = [set1,set2,set3,set4,set5]
cnt = 1
for item in file:
    with open('data'+str(cnt)+'.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data
        for data in item:
            #还原数据
            u1 = float(data[0:8])
            u2 = float(data[8:16])
            v1 = float(data[16:24])
            v2 = float(data[24:])
            writer.writerow([u1,u2,v1,v2])
        cnt+=1

time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum)
for i in range(5):
    with open("dst.csv","a+",newline='') as f_1:
        f_1_csv=csv.writer(f_1)
        with open("data"+str(i+1)+".csv","r+") as f:
            f_csv=csv.reader(f)
            next(f_csv)
            for row in f_csv:
                if row == []:
                    continue
                f_1_csv.writerow(row)
            f_1.close()
        f.close()

# a = 3
# t = Timer(5.0, judge,([a,1,1,1,1],flag))
# t.start() # 5秒后, "hello, world"将被打印
# while True:  
#     # schedule.every(5).seconds.do(judge,a,1,1,1,1)
#     a+=1
#     if flag:
#         break
# # print(str(0.123456)[2:])

# quitJudge = False
# def job(job_name,age):
#     global quitJudge
#     print("I'm working on " + job_name+" age:"+str(age))
#     if age>5 :
#         quitJudge = True
# age = 1
# def add():
#     global age
#     age+=1
# schedule.every(5).seconds.do(job,'lxp',age)
# schedule.every(2).seconds.do(add)
# while True:
    
#     schedule.run_pending() # 运行所有可运行的任务
#     if quitJudge:
#         break
