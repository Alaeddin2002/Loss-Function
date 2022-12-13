#!/usr/bin/env python
# coding: utf-8

# # Data Science: MAE an MSE 
# #### By: Javier Orduz
# <!--
# <img
# src="https://jaorduz.github.io/images/Javier%20Orduz_01.jpg" width="50" align="center">
# -->
# 
# [license-badge]: https://img.shields.io/badge/License-CC-orange
# [license]: https://creativecommons.org/licenses/by-nc-sa/3.0/deed.en
# 
# [![CC License][license-badge]][license]  [![DS](https://img.shields.io/badge/downloads-DS-green)](https://github.com/Earlham-College/DS_Fall_2022)  [![Github](https://img.shields.io/badge/jaorduz-repos-blue)](https://github.com/jaorduz/)  ![Follow @jaorduc](https://img.shields.io/twitter/follow/jaorduc?label=follow&logo=twitter&logoColor=lkj&style=plastic)

# In[1]:


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from scipy import stats as st


# In[2]:


df = pd.read_csv("FuelConsumption.csv")


# In[3]:


df.describe()


# In[4]:


np.median(df.ENGINESIZE)


# In[5]:


np.mean(df.ENGINESIZE)


# In[6]:


st.mode(df.ENGINESIZE)[0][0]


#  We find the MAE when we set 𝜃 to the mode, median, and mean

# In[7]:


def mae_loss(theta, y_vals):
    return np.mean(np.abs(y_vals - theta))
def mse_loss(theta, y_vals):
    return np.mean((y_vals - theta) ** 2)


# In[8]:


thetas = np.arange(-1, 10, 0.1)
y_vals=np.array(df['ENGINESIZE'])

losses = [mae_loss(theta, y_vals) for theta in thetas]

plt.figure(figsize=(4, 3))
plt.plot(thetas, losses)
plt.axvline(np.median(y_vals), linestyle='--',label='Median')
plt.title(r'Mean Absolute Error for Engine Size')
plt.xlabel(r'$ \theta $ Values')
plt.ylabel('Loss');
plt.legend();


# In[9]:


thetas = np.arange(-1, 10, 0.1)
y_vals=df['ENGINESIZE']
losses = [mse_loss(theta, y_vals) for theta in thetas]

plt.figure(figsize=(4, 2.5))
plt.plot(thetas, losses)
plt.axvline(np.mean(y_vals), linestyle='--',label=rf'Mean')
plt.title(r'Mean Squared Error for Engine Size')
plt.xlabel(r'$ \theta $ Values')
plt.ylabel('Loss')
plt.legend();


# One feature of this curve that is quite noticeable is how rapidly large the MSE grows compared to  the MAE (note the range on the vertical axis). This growth has to do with the nature of squaring errors; it places a much higher loss on data values further away. If $\theta = 10$ and $y = 110$, the squared loss is $(10 - 110)^2 = 10000$ whereas the absolute loss is $|10 - 110| = 100$. For this reason, MSE is more sensitive to unusually large values than MAE.

# # Exercises
# 
# 1. Implement a new data base, calculate the MSE, MAE for different attributes. 
# 1. Submmit your report in Moodle. Template https://www.overleaf.com/read/xqcnnnrsspcp

# In[29]:


df = pd.read_csv("drinks.csv")
df.describe()

thetas = np.arange(-1, 10, 0.1)
y_vals=np.array(df['wine_servings'])

losses = [mae_loss(theta, y_vals) for theta in thetas]

plt.figure(figsize=(4, 3))
plt.plot(thetas, losses)
plt.axvline(np.median(y_vals), linestyle='--',label='Median')
plt.title(r'Mean Absolute Error for wine_servings')
plt.xlabel(r'$ \theta $ Values')
plt.ylabel('Loss');
plt.legend();


# In[30]:


thetas = np.arange(-1, 10, 0.1)
y_vals=df['total_litres_of_pure_alcohol']
losses = [mse_loss(theta, y_vals) for theta in thetas]

plt.figure(figsize=(4, 2.5))
plt.plot(thetas, losses)
plt.axvline(np.mean(y_vals), linestyle='--',label=rf'Mean')
plt.title(r'Mean Squared Error for total_litres_of_pure_alcohol')
plt.xlabel(r'$ \theta $ Values')
plt.ylabel('Loss')
plt.legend();


# # References
# 
# [0] data https://tinyurl.com/2m3vr2xp
# 
# [1] numpy https://numpy.org/
# 
# [2] scipy https://docs.scipy.org/
# 
# [3] matplotlib https://matplotlib.org/
# 
# [4] matplotlib.cm https://matplotlib.org/stable/api/cm_api.html
# 
# [5] matplotlib.pyplot https://matplotlib.org/stable/api/pyplot_summary.html
# 
# [6] pandas https://pandas.pydata.org/docs/
# 
# [7] seaborn https://seaborn.pydata.org/
# 
# [8] Data Science: https://www.textbook.ds100.org/intro.html
# 
# [9] Jaccard https://tinyurl.com/27bboh2u
# 
# [10] IBM course. Author: Saeed Aghabzorgi. IBM lab skills. Watson Studio.
# 
# 

# 
