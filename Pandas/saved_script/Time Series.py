#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np


# #### string to time

# In[6]:


#可以接受array作为参数！
help(pd.to_datetime)


# In[7]:


from dateutil.parser import parse
help(parse)


# #### time series basic

# In[16]:


dates = pd.date_range(start='2000/1/1', periods=100, freq='W-WED')
long_df = pd.DataFrame(np.random.randn(100, 4),
                      index=dates,
                      columns=['Colorado', 'Texas',
                               'New York', 'Ohio'])


# In[18]:


#你可以直接选择年月, 返回所有匹配的日期
long_df.loc['2001-5']


# In[20]:


#a useful shorthand for boolean indexing based on index values above or below certain thresholds.
help(long_df.truncate)


# In[21]:


#如何处理重复日期
dates = pd.DatetimeIndex(['1/1/2000', '1/2/2000', 
                          '1/2/2000','1/2/2000', '1/3/2000'])
dup_ts = pd.Series(np.arange(5), index=dates)


# In[24]:


#判断是否有重复
dup_ts.index.is_unique


# In[26]:


#有重复
dup_ts['1/2/2000']


# In[27]:


#去重方法之一: groupby(level=0)
dup_ts.groupby(level=0).mean()


# #### Date Ranges, Frequencies, and Shifting

# In[50]:


#产生time_range
#!!!!!你最多只能指定四个参数中的三个: start, end, freq, periods
pd.date_range(start='2012-04-01', end='2012-06-01', freq='5D')


# In[30]:


N = 150
times = pd.date_range(start='2017-05-20 00:00', freq='1min', periods=N)
df = (pd.DataFrame({'time': times,
                   'value': np.arange(N)})
     .set_index('time'))


# In[51]:


#resample method
#每隔5分钟采样一次
df.resample('5min').mean().head()


# In[38]:


df2 = (pd.DataFrame({'time': times.repeat(3),
                    'key': np.tile(['a', 'b', 'c'], N),
                    'value': np.arange(N * 3.)})
      .set_index('time'))


# In[45]:


#通过pd.TimeGrouper对象来resampling 'key'
#记住了你的索引必须是time！！！
df2.groupby(['key', pd.Grouper(freq='5min')]).sum().head()


# In[ ]:




