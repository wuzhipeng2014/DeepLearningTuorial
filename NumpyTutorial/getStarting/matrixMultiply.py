# encoding:utf-8

import numpy as np

print("矩阵乘法测试")

x = np.array([[1, 2],[3, 4]], dtype=np.float64)

y = np.array([[5, 6],[7, 8]], dtype=np.float64)


print(x[1][1])

print("x="+str(x))

print("y="+str(y))


## 矩阵元素对应相乘

xy=np.multiply(x,y)

print("np.multiply(x,y)="+str(xy))


## 矩阵乘法

print("np.dot(x,y)="+str(np.dot(x,y)))

z=np.dot(x,y)

print("矩阵乘法")
print("np.dot(x,y)="+str(z))



s=np.sum(z,axis=1)

print("np.sum(z)=",s)



## numpy的广播机制


v=np.array([1,2,3])
w=np.array([4,5])

print("np.broadcasting 机制求内积")

print(np.reshape(v,(3,1)) *w)

x=np.array([[1,2,3],[4,5,6]])

print("braodcastring机制求和")
print(x+v)



print (x + np . reshape(w, (2, 1)))











