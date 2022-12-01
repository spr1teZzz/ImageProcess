import urllib.request
import os

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
print(df_data.describe())

