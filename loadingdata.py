import numpy as np
xy=np.loadtxt('test.csv',delimiter=',',dtype=np.float32)
x_data=xy[:0:-1]
y_data1=xy[:,[-1]]
y_data2=xy[:,-1]
print("xdata")
print(x_data.shape,x_data,len(x_data))
print("ydata1")
print(y_data1.shape,y_data1)
print("y_data2")
print(y_data2.shape,y_data2)