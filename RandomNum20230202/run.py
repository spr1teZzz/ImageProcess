import random  
import csv
from threading import Timer
import time
import schedule
import os
SIZE = 1000000#生成的组数

def decimal_digits(val: float):
    #判断小数点后的长度
    val_str = str(val)
    digits_location = val_str.find('.')
    if digits_location:
        return len(val_str[digits_location + 1:])

for i in range(5):      
    set1 = set()#存储U+ 0.0-0.2
    set2 = set()#存储U+ 0.2-0.4
    set3 = set()#存储U+ 0.4-0.6
    set4 = set()#存储U+ 0.6-0.8
    set5 = set()#存储U+ 0.8-1.0
    time_start = time.time()  # 记录开始时间
    while True:
        num = random.randint(3,6)#随机生成3-6位小数
        u1 = round(random.random(),num)
        u2 = round(random.random(),num)
        v1 = round(random.random(),num)
        v2 = round(random.random(),num)
        digit = decimal_digits(u1) + decimal_digits(u2) + decimal_digits(v1) + decimal_digits(v2)
        if  digit<= 24 and digit>=12 and v2<v1 and u2<u1 and u1+v1<=1:
            #将u、v补成指定长度6位，便于后续分割
            u1 = '%.6f'%u1
            u2 = '%.6f'%u2
            v1 = '%.6f'%v1
            v2 = '%.6f'%v2
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
        if len(set1)+len(set2)+len(set3)+len(set4)+len(set5)>=SIZE:#当组数大于SIZE时跳出
            break

    #写入多个文件
    header = ['U+','U-','V+','V-']
    file = [set1,set2,set3,set4,set5]
    with open('data\\data'+str(i)+'.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        for item in file:
            # write the data
            for data in item:
                #还原数据
                u1 = float(data[0:8])
                u2 = float(data[8:16])
                v1 = float(data[16:24])
                v2 = float(data[24:])
                writer.writerow([u1,u2,v1,v2])
            

    time_end = time.time()  # 记录结束时间
    time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
    print(time_sum)
    with open("time.txt","a") as f:
        f.write(""+str(i)+':'+str(time_sum)+'\n')  # 自带文件关闭功能，不需要再写f.close()

# #将多个文件合成一个
# for i in range(5):
#     with open('data\\data'+str(i)+'.csv', 'w', encoding='UTF8', newline='') as f_1:
#         f_1_csv=csv.writer(f_1)
#         f_1_csv.writerow(header)
#         for j in range(5):
#             with open('data\\data'+str(i)+'_'+str(j)+'.csv',"r+") as f:
#                 f_csv=csv.reader(f)
#                 next(f_csv)
#                 for row in f_csv:
#                     if row == []:
#                         break
#                     f_1_csv.writerow(row)
#             f.close()
#         f_1.close()

# #删除多余文件
# # os.remove(path)
# for i in range(5):
#     for j in range(5):
#         os.remove('data\\data'+str(i)+'_'+str(j)+'.csv')