import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import time
#from sklearn.preprocessing import StandardScaler

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 20:45:39 2023

@author: xhackax47
"""

def chargement():
    """Chargement et traitement des données du dataset Fashion MNIST"""
    fashion_mnist = tf.keras.datasets.fashion_mnist
    
    (entrainement_images, entrainement_cibles), (test_images, test_cibles) = tf.keras.datasets.fashion_mnist.load_data()
    
    print(entrainement_images.shape)
    print(entrainement_cibles.shape)
    plt.imshow(entrainement_images[15000]) # affiche la 15000eme image