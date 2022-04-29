from enum import Enum
import numpy
import napari
from napari.types import ImageData
import cv2
from magicgui import magicgui
from skimage import data
import sys

#Etape 1
import numpy
import tensorflow as tf
from tensorflow import keras
#!pip install focal_loss
from focal_loss import BinaryFocalLoss
from tensorflow.keras import backend as K
import numpy as np
import cv2
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info

# here's the magicgui!  We also use the additional
# `call_button` option
@magicgui(call_button="executee")
def image_segmentation(
    layer: ImageData
    ) -> ImageData:
    
    def redimension(image):
        X = np.zeros((1,256,256,3),dtype=np.uint8)
        X[0] = resize(image, (256, 256), mode='constant', preserve_range=True)
        return X

    def dice_coefficient(y_true, y_pred):
        eps = 1e-6
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + eps) #eps pour éviter la division par 0 
    
    
    
    image_reshaped = redimension(layer) #numpy.ndarray
    
    #Etape 3 : ii) Segmentation de l'image par l'algorithme U-NET pré-entrainé
    model_new = tf.keras.models.load_model("best_model_FL_BCE_0_5.h5",custom_objects={'dice_coefficient': dice_coefficient})
    prediction = model_new.predict(image_reshaped)
    #Etape 3 : iii) Application du seuil de segmentation optimisé
    preds_test_t = (prediction > 0.30000000000000004).astype(np.uint8)
    
    #preds_test_t = resize(preds_test_t, (3024, 4032), mode='constant', preserve_range=True)
    #Etape 4 : Output
    temp=np.squeeze(preds_test_t[0,:,:,0])*255
    return cv2.resize(temp, dsize=(3024, 4032))
    #return preds_test_t*255

# create a viewer and add a couple image layers
viewer = napari.Viewer()
viewer.add_image(data.astronaut(), rgb=True)

# add our new magicgui widget to the viewer
viewer.window.add_dock_widget(image_segmentation)

# keep the dropdown menus in the gui in sync with the layer model
viewer.layers.events.inserted.connect(image_segmentation.reset_choices)
viewer.layers.events.removed.connect(image_segmentation.reset_choices)

napari.run()

#il reste à corriger la taille du output et chercher pourquoi l'output n'est pas similaire aux autres output trouvés pécédement
