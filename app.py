import streamlit as st
import numpy as np
from pandas import DataFrame
from flair.embeddings import TransformerDocumentEmbeddings
import seaborn as sns
from functionforDownloadButtons import download_button
import onnxruntime as rt
from fen_to_array import *
import pickle
import pandas as pd
import base64
import chess
import chess.svg

#sess = rt.InferenceSession("model/chess-resnet.onnx")

sess = rt.InferenceSession("model/chess-resnet.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name


def fentovsg(fen):
    board = chess.Board(fen)
    svg = chess.svg.board(board=board)
    return svg

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s" width="500" height="500"/>' % b64
    # adjust height and width
    #html = html.replace('width="8"', 'width="50%"')
    st.write(html, unsafe_allow_html=True)

scaler = pickle.load(open("model/mm_scaler_untilmove80_clipped500.pkl", "rb"))
def rcnn_chess(fenstring='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'): 
    # check if fenstring has only one line
    if fenstring.count('\n') > 0:
        # get list of fenstrings
        fenstrings = fenstring.split('\n')
        # get list of predictions
        predictions = []
        for fenstring in fenstrings:
            #print(fenstring)
            sample = fenToVec(fenstring)
            #print(sample)
            sample = np.expand_dims(sample, axis=0)
            scores = sess.run([output_name], {input_name: sample})[0][0]
            scores = np.array(scores).reshape(1,-2)
            #print(scores)
            prediction = scaler.inverse_transform(scores)
            #print(prediction)
            prediction = np.where(prediction[0][0] > 0, prediction[0][0].item(), 0)
            predictions.append(prediction)
            #print(predictions)
        return predictions
    else:
        sample = fenToVec(fenstring)
        # expand dims
        sample = np.array(np.expand_dims(sample, axis=0))
        scores = sess.run(None, {input_name: sample})[0][0]
        #print(scores)
        scores = np.array(scores).reshape(1,-1)
        prediction = scaler.inverse_transform(scores)
        #print(prediction)
        prediction = np.where(prediction[0][0] > 0, prediction[0][0].item(), 0)
        #print(prediction)
        return prediction

st.set_page_config(
    page_title="Chess Difficulty Prediction",
    page_icon="🎈",
)


def _max_width_():
    max_width_str = f"max-width: 1400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


_max_width_()

c30, c31, c32 = st.columns([2.5, 1, 3])

with c30:
    # st.image("logo.png", width=400)
    st.title("♟️ Chess ")
    st.header("")



with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
-   This tool predicts the *difficulty*, for a given chess position.
- It uses [FEN-Strings](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) to represent chess positions.
- The model is trained is a residual convolutional neural network, trained on approx. 2 mio. *professional* human chess games.
- The Output is a single number, which yields the *difficulty* of the position, measured in Expected Loss of CP (Centipawns).
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("## **📌 Paste FEN-String **")

with st.form(key="my_form"):


    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])

    with c2:
        doc = st.text_area(
            "Paste your FEN-String below (max 10 at once; separate multiple FEN-Strings with a newline):",
            height=110,
        )

        MAX_WORDS = 500
        import re
        res = len(re.findall(r"\w+", doc))
        if res > MAX_WORDS:
            st.warning(
                "⚠️ Your text contains "
                + str(res)
                + " words."
                + " Only the first 500 words will be reviewed. Stay tuned as increased allowance is coming! 😊"
            )

            doc = doc[:MAX_WORDS]

        submit_button = st.form_submit_button(label="✨ Predict Difficulty!")

    
try:
    keywords = rcnn_chess(doc)
    print(keywords)
    print(type(keywords))
    if type(keywords) == list:
        str_keywords = [str(keyword.item()) + "/n" for keyword in keywords]
        print(str_keywords)
    else:
        str_keywords = str(keywords) + "/n"
        print(str_keywords)
except:
    keywords = "No keywords found"
    str_keywords = "Please provide a valid FEN-String"

st.markdown('### Board')
if type(doc) == list:
    render_svg(fentovsg(doc[0]))
else:
    render_svg(fentovsg(doc))

st.markdown("## **🎈 Check & download results **")

st.write(str_keywords)

st.header("")

cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

with c1:
    CSVButton2 = download_button(keywords, "Data.csv", "📥 Download (.csv)")
with c2:
    CSVButton2 = download_button(keywords, "Data.txt", "📥 Download (.txt)")
with c3:
    CSVButton2 = download_button(keywords, "Data.json", "📥 Download (.json)")

st.header("")

df = DataFrame({"FEN-String":[doc],"Difficulty":[keywords]}).sort_values(by="Difficulty", ascending=False)#.reset_index(drop=True)
df.index += 1

# Add styling
cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)
df = df.style.background_gradient(
    cmap=cmGreen,
    subset=[
        "Difficulty",
    ],
)

c1, c2, c3 = st.columns([1, 3, 1])

format_dictionary = {
    "Difficulty": "{:.1%}",
}

df = df.format(format_dictionary)

#with c2:
#    st.table(df)
