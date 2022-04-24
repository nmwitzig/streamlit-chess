#%%
import onnx
import pickle
import numpy as np
from fen_to_array import *
import onnxruntime as rt
model = onnx.load("model/chess-resnet.onnx")

#sess = rt.InferenceSession("model/chess-resnet.onnx")

sess = rt.InferenceSession("model/chess-resnet.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name


scaler = pickle.load(open("model/mm_scaler_untilmove80_clipped500.pkl", "rb"))
def rcnn_chess(fenstring): 
    sample = np.array(fenToVec(fenstring))
    scores = sess.run(None, {input_name: sample})[0][0]
    scores = np.array(scores).reshape(1,-1)
    prediction = scaler.inverse_transform(scores)
    prediction = np.where(prediction[0][0] > 0, prediction[0][0].item(), 0)
    return prediction


#%%
import chess

#get fen of base board
board = chess.Board()
fen = board.fen()

pred = rcnn_chess(fen)
pred
# %%
