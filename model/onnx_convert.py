#%%
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model
import onnx
import keras2onnx

import tensorflow as tf
model = tf.keras.models.load_model('../visualize_training/tb_logs_inputminmax_untilmove80_clipped500ResNet4096.hdf5')
tf.saved_model.save(model, "tmp_model")

#%%
# https://stackoverflow.com/questions/66560370/why-keras2onnx-convert-keras-function-keeps-getting-me-error-kerastensor-ob
# Convert in bash:
!python -m tf2onnx.convert --saved-model tmp_model --output "chess-resnet.onnx"
