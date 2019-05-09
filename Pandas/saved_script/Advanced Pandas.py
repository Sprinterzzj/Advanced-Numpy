#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# ### 类别数据

# ![tupian](img/categorical_methods.png)

# #### 类别数据基础
# 类别数据的元素是不可变类型

# In[3]:


fruits = ['apple', 'orange', 'apple', 'orange'] * 2
N = len(fruits)
df = pd.DataFrame({
    'fruit':fruits,
    'basket_id': np.arange(N),
    'count': np.random.randint(3, 16, size = N),
    'weight': np.random.uniform(0, 4, size = N)
})


# In[5]:


df


# ###### convert to categorical

# In[14]:


fruit_cat = df['fruit'].astype('category')
print(fruit_cat)
#你也可用pd.Categorical
my_categorical = pd.Categorical(['foo', 'bar', 'baz', 'foo'])
print(my_categorical)


# In[20]:


#categorical对象的属性
c = fruit_cat.values
print(type(c))
print(c)
print(c.categories)
print(c.codes)


# ###### 从categories和codes中恢复数据

# In[18]:


categories = ['foo', 'bar', 'baz']
codes = [0, 1, 2, 0, 0, 1]
#用pd.Categorical.from_codes, 通过ordered参数决定是否有顺序
print(pd.Categorical.from_codes(codes, categories, ordered=True))
#你可以为没有顺序的categorical增加顺序
my_categorical.as_ordered()


# #### 用类别数据计算

# In[28]:


np.random.seed(12345)
draws = np.random.randn(1000)
bins = pd.qcut(x=draws, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
print(bins.codes[:5])
print(bins)

#use categorical with groupby
bins = pd.Series(bins, name='quartile')
results = (pd.Series(draws)
          .groupby(bins)
          .agg(['count', 'max', 'min'])
          .reset_index())
print(results)

print(results['quartile'])


# In[34]:


#categorical的两个好处:
#1.性能提升, groupby操作用到categorical上会更快
#2.更少的内存
N = int(10e6)
draws = pd.Series(np.random.randn(N))
labels = pd.Series(['foo', 'bar', 'baz', 'qux'] * (N // 4))
categories = labels.astype('category')
#查看转换前后的内存占用
print(labels.memory_usage())
print(categories.memory_usage())


# In[35]:


#当然转换的过程耗时
get_ipython().run_line_magic('time', "_ = labels.astype('category')")


# #### 一些其他操作

# In[37]:


s = pd.Series(list('abcd') * 2)
cats_s = s.astype('category')
print(cats_s)


# In[39]:


print(cats_s.cat.codes)
print(cats_s.cat.categories)


# In[40]:


#扩展categorical
actual_categories = list('abcde')
cats_s2 = cats_s.cat.set_categories(actual_categories)
print(cats_s.value_counts())
print(cats_s2.value_counts())


# In[41]:


#缩小categorical
cats_s3 = cats_s[cats_s.isin(['a', 'b'])]
cats_s3.cat.remove_unused_categories()


# #### to dummies

# In[45]:


pd.get_dummies(cats_s)


# ### assign and chaining methods

# In[2]:


#我们用之前的例子
fruits = ['apple', 'orange', 'apple', 'orange'] * 2
N = len(fruits)
df = pd.DataFrame({
    'fruit':fruits,
    'basket_id': np.arange(N),
    'count': np.random.randint(3, 16, size = N),
    'weight': np.random.uniform(0, 4, size = N)
})


# In[4]:


df


# #### 用assign方法来新建列

# In[5]:


result = (df.assign(total_weight = df['count'] * df['weight'])
         .groupby('fruit')['total_weight']
         .mean())


# #### 在使用chaining method的时候, 我们无法直接访问“中间变量”

# In[7]:


#比如如下的方法无法连接起来
df_2 = df.loc[0: 5, :]
df_3 = df_2[df_2['count'] >=5]


# In[9]:


#解决方案, 使用匿名函数
result = (df.loc[0:5, :]
         [lambda x: x['count']>=5])


# In[10]:


result


# #### pipe函数
# 1. f(df, arg = a) 等价于 df.pipe(f, arg = a)
# 2. pipe函数顾名思义是可以chain的

# In[ ]:




