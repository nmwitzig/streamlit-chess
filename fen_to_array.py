#%%
import re
import chess
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import ast




# Generate the regexs
boardRE = re.compile(r"(([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)) ((w)|(b)) ((-)|(K)?(Q)?(k)?(q)?)( ((-)|(\w+)))?( \d+)?( \d+)?")

replaceRE = re.compile(r'[1-8/]')

pieceMapWhite = {'E' : [False] * 12}
pieceMapBlack = {'E' : [False] * 12}

piece_reverse_lookup = {}

all_pieces = 'PNBRQK'

for i, p in enumerate(all_pieces):
    #White pieces first
    mP = [False] * 12
    mP[i] = True
    pieceMapBlack[p] = mP
    piece_reverse_lookup[i] = p

    #then black
    mP = [False] * 12
    mP[i + len(all_pieces)] = True
    pieceMapBlack[p.lower()] = mP
    piece_reverse_lookup[i  + len(all_pieces)] = p.lower()


    #Black pieces first
    mP = [False] * 12
    mP[i] = True
    pieceMapWhite[p.lower()] = mP

    #then white
    mP = [False] * 12
    mP[i + len(all_pieces)] = True
    pieceMapWhite[p] = mP

iSs = [str(i + 1) for i in range(8)]
eSubss = [('E' * i, str(i)) for i in range(8,0, -1)]
castling_vals = 'KQkq'

def toByteBuff(l):
    return b''.join([b'\1' if e else b'\0' for e in l])

pieceMapBin = {k : toByteBuff(v) for k,v in pieceMapBlack.items()}

def toBin(c):
    return pieceMapBin[c]

castlesMap = {True : b'\1'*64, False : b'\0'*64}

#Some previous lines are left in just in case

# using N,C,H,W format


def simple_fen_vec(boardStr, is_white, castling, en_passant):
    castles = [np.frombuffer(castlesMap[c], dtype='bool').reshape(1, 8, 8) for c in castling]
    board_buff_map = map(toBin, boardStr)
    board_buff = b''.join(board_buff_map)
    a = np.frombuffer(board_buff, dtype='bool')
    a = a.reshape(8, 8, -1)
    a = np.moveaxis(a, 2, 0)
    if is_white:
        colour_plane = np.ones((1, 8, 8), dtype='bool')
    else:
        colour_plane = np.zeros((1, 8, 8), dtype='bool')
    if en_passant:
        enpassant_plane = np.ones((1, 8, 8), dtype='bool')
    else:
        enpassant_plane = np.zeros((1, 8, 8), dtype='bool')

    return np.concatenate([a, colour_plane, enpassant_plane, *castles], axis = 0)

def preproc_fen(fenstr):
    r = boardRE.match(fenstr)
    if r.group(14):
        castling = (False, False, False, False)
    else:
        castling = (bool(r.group(15)), bool(r.group(16)), bool(r.group(17)), bool(r.group(18)))
    if r.group(11):
        is_white = True
        rows_lst = r.group(1).split('/')
    else:
        is_white = False
        castling = castling[2:] + castling[:2]
        rows_lst = r.group(1).swapcase().split('/')
        rows_lst = reversed([s[::-1] for s in rows_lst])

    rowsS = ''.join(rows_lst)
    for i, iS in enumerate(iSs):
        if iS in rowsS:
            rowsS = rowsS.replace(iS, 'E' * (i + 1))

    if r.group(20) == "-":
        en_passant = False
    else:
        en_passant = True
    return rowsS, is_white, castling, en_passant

def fenToVec(fenstr):
    previous_output = simple_fen_vec(*preproc_fen(fenstr))
    t = previous_output.transpose(1,2,0)
    # add one dimension for batch size
    #t = np.expand_dims(t, axis=0)
    #t = np.expand_dims(t,axis=0)
    t = tf.cast(t, tf.float32)
    return t
