import urllib.request
import os
from sklearn import preprocessing
data_url = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanice.xls"
data_file_path ="D:\\VScodeProject\\pythonProject\\TensorFlowDemo\\Demo\\KerasApplication\\titanic3.xls"

if not os.path.isfile(data_file_path):
    result = urllib.request.urlretrieve(data_url,data_file_path)
    print("downloaded:",result)
else:
    print(data_file_path,"data file already exits.")

import numpy
import pandas as pd
#读取数据文件，结果为DataFrame格式
df_data = pd.read_excel(data_file_path)

#查看数据摘要
# print(df_data.describe())
#筛选提取需要的特征字段,去掉ticket,cabin
selected_cols = ["survived","name","pclass","sex","age","sibsp","parch","fare","embarked"]
selected_df_data = df_data[selected_cols]
# print(selected_df_data.isnull().any())
# print(selected_df_data[selected_df_data.isnull().values==True])

#为缺失age记录填充 设置为平均值
age_mean_value = selected_df_data["age"].mean()
selected_df_data["age"] = selected_df_data["age"].fillna(age_mean_value)

#为确实fare记录填充值
fare_mean_value = selected_df_data["fare"].mean()
selected_df_data["fare"] = selected_df_data["fare"].fillna(fare_mean_value)

#为确实embarked记录填充值
selected_df_data["embarked"] = selected_df_data["embarked"].fillna("S")

#转换编码
#性别sex由字符串转为数字编码
selected_df_data["sex"] = selected_df_data["sex"].map({"female":0,"male":1}).astype(int)

#港口embarked由字母表示转换为数字编码
selected_df_data["embarked"] = selected_df_data["embarked"].map({"C":0,"Q":1,"S":2}).astype(int)

#删除name字段
#drop不改变原有的df中的数据，而是返回另一个DataFrame来存放删除后的数据axis =1 表示删除列
selected_df_data = selected_df_data.drop(["name"],axis=1)
# print(selected_df_data[:3])

#分离特征值和标签值
#转化为ndarray数组
ndarray_data = selected_df_data.values
#后7列是特征值
features = ndarray_data[:,1:]
#第0列是标签值
label = ndarray_data[:,0]

#特征值标准化
minmax_scale = preprocessing.MinMaxScaler(feature_range=(0,1))
norm_features = minmax_scale.fit_transform(features)

print(norm_features[:3])