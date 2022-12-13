#!/usr/bin/env python
# coding: utf-8

# # Data Science: Logistic Regression 
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
# 

# In[2]:


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[3]:


def mae_loss(theta, y_vals):
    return np.mean(np.abs(y_vals - theta))


# In[19]:


thetas = np.arange(-2, 8, 0.05)
y_vals=np.array([-1, 0, 2, 5, 10])
losses = [mae_loss(theta, y_vals) for theta in thetas]

plt.figure(figsize=(4, 2.5))
plt.plot(thetas, losses)
plt.axvline(np.median(y_vals), linestyle='--',
                    label=rf'Median: $\theta = 2$')
plt.title(r'Mean Absolute Error when $\bf{y}$$ = [-1, 0, 2, 5, 10] $')
plt.xlabel(r'$ \theta $ Values')
plt.ylabel('Loss');
plt.legend();


# In[21]:


def mean_squared_error(act, pred):
    diff = pred - act
    differences_squared = diff ** 2
    mean_diff = differences_squared.mean()
   
    return  mean_diff

thetas=np.array([2])
y_vals=np.array([-1, 0, 2, 5, 10])
print(mean_squared_error(-2,y_vals))
print(mean_squared_error(thetas,y_vals))
print(mean_squared_error(8,y_vals))
print(mean_squared_error(0.05,y_vals))

plt.plot([-2,2, 8],[ 42.8,17.2,38.8])


# # Exercises
# 
# 1. Obtain the Loss function by hand, and compare with Loss function on plots.
# 1. Submmit your report in Moodle. Template https://www.overleaf.com/read/xqcnnnrsspcp

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
