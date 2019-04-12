#!/usr/bin/env python
# coding: utf-8

# # Advanced Numpy

# In[2]:


import numpy as np


# ## ndarray 的结构
# **ndarray** =
# block of memory + indexing scheme + data type descriptor
# 
# * raw data
# * how to locate an element
# * how to interpret an element
# 
# <center>
#     ndarray在内存中的结构<br>
#     
# ![图片](img/threefundamental.png)
# </center>   
# 
# * **memory block**: may be shared, .base, .data
# * **data type descriptor**: structured data, sub-arrays, byte order, casting, viewing, .astype(), .view()
# * **strided indexing**: strides, C/F-order, slicing w/ integers, as_strided, broadcasting, stride tricks, diag, CPU cache coherence
# 
# ```C
# typedef struct PyArrayObject {
#         PyObject_HEAD
# 
#         /* Block of memory */
#         char *data;
# 
#         /* Data type descriptor */
#         PyArray_Descr *descr;
# 
#         /* Indexing scheme */
#         int nd;
#         npy_intp *dimensions;
#         npy_intp *strides;
# 
#         /* Other stuff */
#         PyObject *base;
#         int flags;
#         PyObject *weakreflist;
# } PyArrayObject;```
# 

# ### np.array的若干属性和方法
# * shape 数组的形状
# * dtype 数组元素的types
# * strides 在每个维度中连续两个元素之间的btye数
# * data 数据在内存中的表示
# * \_\_array_interface\_\_ 包含上面属性的字典
# * flags 数据在内存中的信息, 如排列方式, 是否只读等

# In[3]:


arr = np.ones((10, 5), dtype = np.int8)


# In[5]:


#1. shape
print(arr.shape)


# In[6]:


#2. dtype
print(arr.dtype)


# In[29]:


#3. strides
arr.strides
#一般来说 某个维度的stride数目越高, 那么在该维度上计算开销越大


# In[7]:


#4. data
print(bytes(arr.data))


# In[8]:


#全部信息: __array_interface__, flags
print(arr.__array_interface__)


# In[9]:


print(arr.flags)


# #### dytpe
# dtype 的三个属性
# * type
# * itemsize <u>返回单个数据的字节数</u>
# * byteorder

# In[13]:


#type
print(np.dtype(int).type)
print(np.dtype(float).type)
print(np.dtype(np.int8).type)


# In[14]:


#itemsize
print(np.dtype(float).itemsize)


# In[15]:


#byteorder
print(np.dtype(float).byteorder)


# ###### dtype有很多, 但是它们是由几个基类派生来的

# In[19]:


ints = np.ones(10, dtype = np.uint16)
floats = np.ones(10, dtype = np.float32)
#用 np.issubdtype函数来判断一个dtype是不是另外一个dtype的派生
print(np.issubdtype(ints.dtype, np.integer))
print(np.issubdtype(floats.dtype, np.floating))


# In[20]:


#用 mro()方法可以查看一个dtype的所有父类
print(np.float64.mro())


# ##### astype 和 view
# * <u>astype 返回数组的copy 但是 view不会</u>
# * astype recast了数组, 通过改变dtype
# * view只是改变数组在内存中的寻址方式

# In[25]:


x = np.array([1,2,3,4], dtype = np.float32)
#astype recast了array
y = x.astype(np.int8)
#view不会返回copy
z = x.view(np.int8)
#用base函数判断两个数组是否来自同一块内存
y.base is x, z.base is x


# #### order(数组在内存中的存储方式)
# * C: last dimensions vary fastest (= smaller strides)
# * F: first dimensions vary fastest

# In[44]:


#C-order， 按行存储
x = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.int16, order='C')
#F-order， 按列存储
y = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=np.int16, order='F')

print(x.strides)
print(y.strides)

# print(bytes(x.data))
# print(bytes(y.data))


# ## array 基本操作

# ### 索引
# * 切片
# * as_strided
# * fancy indexing: np.take, np.compress

# In[29]:


x = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=np.int8)


# In[34]:


#正常索引, 访问第一行第二列的元素
print(x[1, 2])

#你也可以计算byte offset, 移动指针那样索引
print(x.flat[x.strides[0] * 1 + x.strides[1] * 2])


# ###### slicing返回数组的view

# In[16]:


x = np.array([1,2,3,4,5,6], dtype = np.int32)
y = x[::-1]
print('x:', x)
print('y:', y)
#查看而这是否来自同一块内存
print(y.base is x)
print('stride of x:', x.strides)
print('stride of y:', y.strides)

y = x[2:]
#np.int32占 4个字节, 因此二者内存首地址差了8个字节
print(y.__array_interface__['data'][0] - x.__array_interface__['data'][0])


# ###### fake dimensions with strides
# indexing最终要转换成 byte_offset:<br>
# $byte\_offset = stride[0]*index[0] + stride[1]*index[1] + ...$<br>
# 因此,通过改变strides和shape可以"看上去"改变数组的形状

# In[21]:


from numpy.lib.stride_tricks import as_strided
#help(as_strided)
x = np.array([1,2,3,4], dtype = np.int16)
#as_strided 函数给数组分配新的shape和stride
print(as_strided(x, strides = (2*2, ), shape = (2, )))


# In[42]:


# a little trick, 注意该函数只返回view
arr = np.array([1, 2, 3, 4], dtype=np.int8)
#第一个维度的stride为0
new_arr = as_strided(arr, shape = (3, 4),strides=(0,1))

print(new_arr)
arr[0] = 10
print(new_arr)


# In[19]:


arr = np.arange(10) * 100
print('arr:', arr)
inds = [7, 1, 2, 6]
print('fancy indexing',arr[inds])

#用take来fancy indexing
print('arr.take(inds)',arr.take(inds))

#用 put来fancy assign
#put不接受axis他按照数组在内存中的存储方式来assign
arr.put(inds, 100)
print('after calling arr.put(inds, 100):\n',arr)

arr.put(inds, [40, 41,42,43])
print('after calling arr.put(inds, [40,41,42,43]):\n',arr)

#take也可以用来在其他维度indexing 通过传递axis参数
arr = np.random.randn(2, 4)
inds = [2, 0, 2, 1]
arr.take(inds, axis = 1)
print('arr.take(inds, axis = 1):\n',arr)


# ### reshape
# * arange
# * .T
# * flatten 和 ravel

# In[6]:


arr = np.arange(8)
print('原始数组:\n',arr)

#reshape
print('改变形状后:\n',arr.reshape((4, 2)))
#第二个参数为 -1, 那么按照第一个参数reshape
print('改变形状后:\n',arr.reshape((2, -1)))

#reshape返回copy
#转置返回view(仅仅交换了stride)
print('reshape((4, 2))后转至:\n',arr.reshape((4, 2)).T)

#flatten, 展开数组, 返回copy
print(arr.reshape((2, -1)).flatten())

#ravel 返回一个view
print(arr.reshape((2, -1)).ravel())


# ### array的拼接和分割
# * np.concatenate
# * np.vstack, np.hstack
# * np.r_, np.c_

# In[3]:


#array的拼接
arr1 = np.array([[1,2,3], [4, 5,6]])
arr2 = np.array([[7,8,9], [10, 11, 12]])
print('arr1:\n', arr1)
print('arr2:\n', arr2)

#按行拼接
arr = np.concatenate([arr1, arr2], axis = 0)
print('按行拼接:\n',arr)

#按列拼接
arr = np.concatenate([arr1, arr2], axis = 1)
print('按列拼接:\n',arr)

#vstack 按行拼接
arr = np.vstack((arr1, arr2))
print('按行拼接:\n',arr)

#hstack 按列拼接
arr = np.hstack((arr1, arr2))
print('按列拼接:\n',arr)


# In[5]:


#array的分割
arr = np.random.randn(5, 2)
print('arr:\n', arr)
#arr[0:1,:], arr[1:3,:], arr[3:5,:]
arr1, arr2, arr3 = np.split(arr, [1, 3], axis = 0)
print('arr[:1, :]:\n', arr1)
print('arr[1:3, :]:\n', arr2)
print('arr[3:, :]:\n', arr3)


# ######  拼接函数的简写 np.r_ 与  np.c_

# In[10]:


arr1 = np.arange(6).reshape((3, 2))
arr2 = np.random.randn(3, 2)

arr = np.r_[arr1, arr2]
print('np.r_:按行拼接\n', arr)

np.c_[arr1, arr2]
print('np.c_:按列拼接\n', arr)


# ###### np.c_和np.r_转化切片为array

# In[51]:


np.r_[1:6, -10:-5]


# In[52]:


np.c_[1:6, -10:-5]


# ### 重复数组的元素
# * np.repeat
# * np.tile

# In[15]:


arr = np.arange(3)
print('arr:', arr)

x = arr.repeat(3)
print('arr.repeat(3):', x)

x = arr.repeat([2, 3, 4])
print('arr.repeat([2, 3, 4]):', x)

#注意对多维数组你需要传递axis否则它会先展开再repeat
arr = np.random.randn(2, 2)
print('arr:\n', arr)
print('arr.repeat(2):\n', arr.repeat(2))

x = arr.repeat(2, axis = 0)
print('arr.repeat(2, axis = 0):\n', x)
x = arr.repeat(2, axis = 1)
print('arr.repeat(2, axis = 1):\n', x)
x = arr.repeat([2, 3], axis = 0)
print('arr.repeat([2, 3], axis = 0):\n', x)


# In[16]:


#title用来进行块状重复
x = np.tile(arr, 2)
print('np.tile(2):\n', x)

x = np.tile(arr, (3, 2))
print('np.tile(arr, (3, 2)):\n', x)


# ### 数组广播

# #### 广播计算: 两个数组的 trailing dimension matches 或者 有一个维度是1

# In[31]:


#一维广播计算
arr = np.arange(5)
print('原始数组:', arr)
print('arr * 5:',arr * 5)


# In[33]:


#二维广播计算
arr = np.random.randn(4, 3)
arr.mean(axis = 0)
print('原始数组:\n',arr)
#广播计算, 按行计算
print('按行广播计算:\n',arr - arr.mean(axis = 0))
#广播计算, 按列计算。注意必须让减数的列维度为1
print('按列广播计算:\n',arr - arr.mean(axis = 1)[:, np.newaxis])


# ###### np.newaxis

# In[35]:


arr = np.zeros((4, 4))
print('原始数组:\n', arr)
arr_3d = arr[:, np.newaxis, :]
print('arr[:, np.newaxis, :]\n', arr_3d)
arr_3d = arr[:,:, np.newaxis]
print('arr[:,:, np.newaxis]\n', arr_3d)
arr_3d = arr[np.newaxis,:,:]
print('arr[np.newaxis,:,:]\n', arr_3d)
# arr_1d = np.random.normal(size = 3)

# arr_1d[:, np.newaxis]

# arr_1d[np.newaxis, :]


# ###### 广播计算与 as_strided

# In[21]:


from numpy.lib.stride_tricks import as_strided
x = np.array([1,2,3,4], dtype = np.int16)
x2 = as_strided(x, strides=(0, 1*2), shape = (3, 4))
y = np.array([5,6,7], dtype = np.int16)
y2 = as_strided(y, strides=(1*2, 0), shape = (3, 4))
print('x2:\n', x2)
print('y2:\n', y2)
print('实际上是一个外积运算:\n', x2 * y2)
#等价于如下的广播计算:
print(x[np.newaxis, :] * y[:, np.newaxis])


# ###### 三维的广播计算

# In[107]:


arr = np.random.randn(3, 4, 5)


# In[109]:


#按照最后一个维度取均值
arr.mean(axis = 2)


# In[110]:


#上面的操作等价于
(arr[:, :, 0] + arr[:, :, 1] + arr[:, :, 2] + arr[:, :, 3] + arr[:, :, 4]) / 5


# In[111]:


#按照中间的维度取平均值
arr.mean(axis = 1)


# In[114]:


#对arr的每一页, 取第零行的长度为5的数组
arr[:, 0, :]


# In[115]:


#注意最后的维度必须是1
arr - arr.mean(axis = 2)[:, :, np.newaxis]


# ###### np.newaxis 返回copy, 下面的方法不损耗性能

# In[129]:


indexer = [slice(None)] * arr.ndim


# In[130]:


indexer


# In[131]:


indexer[1] = np.newaxis


# In[132]:


#这样不生成copy
arr - arr.mean(1)[tuple(indexer)]


# #### 广播赋值

# In[133]:


arr = np.zeros((4, 3))


# In[134]:


arr[:] = 5


# In[135]:


arr


# In[137]:


col = np.array([1, 2, 3, -4])
arr[:] = col[:, np.newaxis]


# In[139]:


arr


# In[142]:


arr[:2, :] = np.array([[1.37], [0.5]])


# In[143]:


arr[:2, :]


# In[144]:


np.array([[1.37], [0.5]])


# ### Numpy ufunc

# In[145]:


#reduce
arr = np.arange(10).reshape((5, -1))


# In[146]:


arr


# In[147]:


np.add.reduce(arr, axis = 0)


# In[148]:


np.add.reduce(arr, axis = 1)


# In[151]:


arr = np.random.randn(5, 5)
arr[::2, :].sort(1)


# In[152]:


arr[:, :-1]< arr[:, 1:]


# In[153]:


np.logical_and.reduce(arr[:, :-1]< arr[:, 1:], axis = 1)


# In[154]:


#accumulate
arr = np.arange(15).reshape((3, -1))


# In[156]:


arr


# In[157]:


#相当于np.cumsum
np.add.accumulate(arr, axis = 1)


# In[161]:


#outer,外积运算
arr = np.arange(3).repeat([1, 2, 2])


# In[162]:


arr


# In[163]:


np.multiply.outer(arr, np.arange(5))


# In[168]:


#outer结果的维度是两个输入维度的笛卡尔积
x, y = np.random.randn(3, 4), np.random.randn(5)


# In[165]:


result = np.subtract.outer(x, y)


# In[167]:


result.shape


# In[169]:


#reduceat "local reduce"或者说groupby reduce


# In[171]:


#sum on arr[0:5], arr[5:8], arr[8:]
arr = np.arange(10)
np.add.reduceat(arr, [0, 5, 8])


# In[172]:


arr = np.multiply.outer(np.arange(4), np.arange(5))


# In[173]:


arr


# In[174]:


np.add.reduceat(arr, [0, 2, 4], axis = 1)


# ### 定义自己的 ufunc

# In[175]:


#这需要用到 numpy的 C-api


# In[177]:


#你也可以用 np.vectorize

def add_elements(x, y):
    return x + y
add_them = np.vectorize(add_elements, otypes = [np.float64])


# #### 但是速度会变慢！

# In[178]:


get_ipython().run_line_magic('time', 'add_them(np.arange(8), np.arange(8))')


# In[179]:


get_ipython().run_line_magic('time', 'np.add(np.arange(8), np.arange(8))')


# ### 结构数组

# In[180]:


dtypes = [('x', np.float64), ('y', np.int32)]
s_arr = np.array([(1.5, 6), (np.pi, -2)], dtype = dtypes)


# In[181]:


s_arr


# In[182]:


s_arr['x']


# In[188]:


#每一个tuple代表一行, 其中每个元素有自己的名字和dtype
s_arr = np.array([(1, 2), (3, 4)],
                 dtype = [('foo', np.int32), ('bar', np.float16)])


# In[189]:


s_arr['foo']


# In[190]:


#你可以指定每个名字的元素数目，即nested dtypes
dtypes = [('x', np.int64, 3), ('y', np.int32)]
arr = np.zeros(4, dtype = dtypes)


# In[192]:


#按元素名字访问
arr['x']


# In[194]:


#按行访问
arr[0]


# In[195]:


#你也可以指定 nested array的dtype
dtype = [('x', [('a', 'f8'), ('b', 'f4')]), ('y', np.int32)]
data = np.array([((1, 2), 5), ((3, 4), 6)], dtype = dtype)


# In[196]:


data


# In[198]:


data['x']


# In[199]:


data['y']


# ### 数组排序

# In[200]:


arr = np.random.randn(6)


# In[201]:


arr


# In[204]:


#inplace sort
arr.sort()


# In[205]:


arr


# In[214]:


#注意排序是 in-place的
arr = np.random.randn(3, 5)
print(arr)
#arr[:, 0]这是view,因而会改变原数组
arr[:, 0].sort()


# In[215]:


arr


# #### 与arr.sort()不同, np.sort返回copy

# In[216]:


arr = np.random.randn(5)


# In[217]:


arr


# In[218]:


np.sort(arr)


# In[219]:


#上面的两种sort函数都可以传递axis
arr = np.random.randn(3, 5)


# In[220]:


arr


# In[221]:


#按列排序
np.sort(arr, axis = 1)


# In[223]:


#按行排序
np.sort(arr, axis = 0)


# #### 注意上面的排序都只产生升序序列, 在实际中 用[::-1]来得到数组的逆序

# In[224]:


arr


# In[225]:


arr[:, ::-1]


# #### 间接排序 argsort lexsort

# In[226]:


values = np.array([5, 0, 1, 3, 2])
indexer = values.argsort()


# In[227]:


indexer


# In[228]:


values.take(indexer)


# In[233]:


##argsort用于二维数组


# In[229]:


arr = np.random.randn(3, 5)
arr[0] = values


# In[230]:


arr


# In[231]:


arr[0]


# In[232]:


arr[:, arr[0].argsort()]


# In[238]:


#下面演示lexsort
#lexsort执行字典排序,注意last_name是第一序!!!!


# In[235]:


first_name = np.array(['Bob', 'Jane', 'Steve', 'Bill', 'Barbara'])
last_name = np.array(['Jones', 'Arnold', 'Arnold', 'Jones', 'Walters'])


# In[236]:


sorter = np.lexsort((first_name, last_name))


# In[237]:


sorter


# In[239]:


last_name[sorter]


# In[240]:


first_name[sorter]


# #### 其他排序算法

# In[241]:


values = np.array(['2:first', '2:second', '1:first',
                   '1:second', '1:third'])
keys = np.array([2, 2, 1, 1, 1])


# In[245]:


#唯一的稳定的排序算法: 归并排序


# In[242]:


indexer = keys.argsort(kind = 'mergesort')


# In[243]:


indexer


# In[244]:


values.take(indexer)


# #### 部分排序算法

# In[246]:


arr = np.random.randn(20)


# In[247]:


arr


# In[249]:


#返回的数组中前三个元素是top3 smallest
np.partition(arr, 3)


# In[254]:


#返回index, 其中前三个索引是top3 smallest的索引
index = np.argpartition(arr, 3)


# In[255]:


index


# In[256]:


np.take(arr, index)


# In[257]:


# np.searchsorted


# In[258]:


arr = np.array([0, 1, 7, 12, 15])


# In[260]:


#顾名思义, 此方法在一个已经排序的数组上
#执行二分搜索, 返回被搜索元素的下标
arr.searchsorted(7)


# In[261]:


arr = np.array([0,0,0,1,1,1,1])


# In[263]:


#返回左起第一个匹配的元素的下标
arr.searchsorted([0, 1])


# In[264]:


#你也可以改变匹配策略
arr.searchsorted([0, 1], side = 'right')


# In[273]:


#一个简单的例子
#这个用法要牢记！！！


# In[268]:


data = np.floor(np.random.uniform(0, 10000, size = 50))


# In[269]:


data


# In[270]:


bins = np.array([0, 100, 1000, 5000, 10000])


# In[276]:


#返回值 3意味着, 2877 在 1000到5000之间
labels = bins.searchsorted(data)


# In[277]:


labels


# In[278]:


import pandas as pd
pd.Series(data).groupby(labels).mean()


# ### 数组输入输出

# In[279]:


#memory-mapped files


# In[282]:


#create a new memory map
mmap = np.memmap('mymap', dtype = np.float64,
                 mode = 'w+', shape = (10000, 10000))


# In[283]:


mmap


# In[284]:


#切片 memmap返回一个view on disk
section = mmap[:5, :]


# In[285]:


section


# In[286]:


#对这个view写入会buffered到memory里
section[:] = np.random.randn(5, 10000)


# In[287]:


#通过flush操作, 写入磁盘中


# In[290]:


mmap.flush()


# In[292]:


del mmap


# In[ ]:




