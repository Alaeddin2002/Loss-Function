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
# 
# 

# In[1]:


import sys
import os
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


def mae_loss(theta, y_vals):
    return np.mean(np.abs(y_vals - theta))


# In[3]:


def try_thetas(thetas, y_vals, xlims, loss_fn=mae_loss, figsize=(5, 4),
               rug_height=0.1, cols=3):
    if not isinstance(y_vals, np.ndarray):
        y_vals = np.array(y_vals)
    rows = int(np.ceil(len(thetas) / cols))
    plt.figure(figsize=figsize)
    for i, theta in enumerate(thetas):
        ax = plt.subplot(rows, cols, i + 1)
        sns.rugplot(y_vals, height=rug_height, ax=ax)
        plt.axvline(theta, linestyle='--',
                    label=rf'$ \theta = {theta} $')
        plt.title(f'Loss = {loss_fn(theta, y_vals):.2f}')
        plt.xlim(*xlims)
        plt.yticks([])
        plt.legend()
    plt.tight_layout()


# In[4]:


try_thetas(thetas=[-2, 0, 1, 2, 3, 4],
           y_vals=[-1, 0, 2, 5, 10],
           rug_height=0.3,
           xlims=(-3, 12))


# # Exercises
# 
# 1. Obtain the Loss function by hand, and compare with Loss function on plots.
# 1. Submmit your report in Moodle. Template https://www.overleaf.com/read/xqcnnnrsspcp

# In[40]:


def mean_squared_error(act, pred):
    diff = pred - act
    differences_squared = diff ** 2
    mean_diff = differences_squared.mean()
   
    return mean_diff

thetas=np.array([-2])
y_vals=np.array([-1, 0, 2, 5, 10])

print(mean_squared_error(thetas,y_vals)/8)
print(mean_squared_error(0,y_vals)/8)
print(mean_squared_error(1,y_vals)/6)
print(mean_squared_error(2,y_vals)/5)
print(mean_squared_error(3,y_vals)/4)
print(mean_squared_error(4,y_vals)/3)


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

# In[ ]:




