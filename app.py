from operator import contains
from requests import session
import streamlit as st
import numpy as np
from pandas import DataFrame
import seaborn as sns
from functionforDownloadButtons import download_button
import onnxruntime as rt
from fen_to_array import *
import pickle
import pandas as pd
import base64
import chess
import chess.svg
import sklearn

# set fen to session state

if "fen" not in st.session_state:
    st.session_state["fen"] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


#sess = rt.InferenceSession("model/chess-resnet.onnx")

sess = rt.InferenceSession("model/chess-resnet.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

def delete_duplicates(df):
    df.drop_duplicates(subset=['fens'], keep='last', inplace=True)
    return df

low_examples = delete_duplicates(pd.read_csv('examples/low.csv', usecols=["fens"]))
high_examples = delete_duplicates(pd.read_csv('examples/high.csv', usecols=["fens"]))
medium_examples = delete_duplicates(pd.read_csv('examples/middle.csv', usecols=["fens"]))


def fentovsg(fen):
    board = chess.Board(fen)
    svg = chess.svg.board(board=board)
    return svg

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s" width="500" height="500"/>' % b64
    # adjust height and width
    html = html.replace("width=\"500\"", "width=\"75%\"")
    html = html.replace("height=\"500\"", "height=\"75%\"")
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
    layout="wide",
)


def _max_width_():
    max_width_str = f"max-width: 1900px;"
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
    st.title("♟️ Chess Difficulty Prediction")
    st.header("")



with st.expander("ℹ️ - About this app", expanded=True):

    st.write(
        """     
- This tool predicts the *difficulty* for a given chess position, measured as a single number in Expected Loss of CP (Centipawns).
- It uses [FEN-Strings](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) to represent chess positions.
- You can use, e.g., the [Lichess Editor](https://lichess.org/editor) to create FEN-Strings.
- The underlying model is a residual convolutional neural network and was trained on moves of approx. 2 mio. evaluated *professional* human chess games. Both the engine evaluations and the training of the model were performed on the [MOGON2-architecture](https://hpc.uni-mainz.de).
- [Please reach out to us](mailto:niklas.witzig@uni-mainz.de) if you have any questions or suggestions.
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("## ** 🎲 Draw or 📌 Paste FEN-String **")
fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


col1, col2, col3,_ = st.columns([1, 1.15, 1, 2.25])
# get st button
with col1:
    if st.button("Draw Low Difficulty example"):
        fen = low_examples.sample(n=1).iloc[0][0]
        st.session_state["fen"] = fen
with col2:
    if st.button("Draw Medium Difficulty example"):
        fen = medium_examples.sample(n=1).iloc[0][0]
        st.session_state["fen"] = fen
with col3:
    if st.button("Draw High Difficulty example"):
        fen = high_examples.sample(n=1).iloc[0][0]
        st.session_state["fen"] = fen

with st.form(key="my_form"):

    c1,_,c3 = st.columns([5,1,5])
    with c1:
        doc = st.text_input(
            "Paste your FEN-String below (max 1 at once)",
            value=st.session_state["fen"],
            key="fen",
            )
        # check for new line
        if doc.count('\n') > 0:
        
            st.warning(
                "⚠️ Your text contains"+
                str(doc.count('\n')) +
                + " FEN Strings."
                + " Only the first will be evaluated"
            )

            doc = doc.split('\n')[0]

        submit_button = st.form_submit_button(label="✨ Predict Difficulty!")
        
        st.markdown(" ## Predicted Difficulty: ")

        try:
            keywords = rcnn_chess(doc)
            if type(keywords) == list:
                str_keywords = [str(keyword.item()) + "/n" for keyword in keywords]
            else:
                str_keywords = keywords.__format__('.2f')
        except:
            keywords = "No keywords found"
            str_keywords = "Please provide a valid FEN-String"
        
        color = "White" if "w" in doc else "Black" 
        st.markdown("**"  + str_keywords + "**" + " Expected CP Loss for " + color)
    


    
    with c3:
        st.markdown('### Corresponding Board')
        if type(doc) == list:
            render_svg(fentovsg(doc[0]))
        else:
            render_svg(fentovsg(doc))
    
    st.write("\n")


#st.markdown("## **🎈 Check & download results **")

#st.write(str_keywords)

# st.header("")

# cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

# with c1:
#     CSVButton2 = download_button(keywords, "Data.csv", "📥 Download (.csv)")
# with c2:
#     CSVButton2 = download_button(keywords, "Data.txt", "📥 Download (.txt)")
# with c3:
#     CSVButton2 = download_button(keywords, "Data.json", "📥 Download (.json)")

# st.header("")

# df = DataFrame({"FEN-String":[doc],"Difficulty":[keywords]}).sort_values(by="Difficulty", ascending=False)#.reset_index(drop=True)
# df.index += 1

# # Add styling
# cmGreen = sns.light_palette("green", as_cmap=True)
# cmRed = sns.light_palette("red", as_cmap=True)
# df = df.style.background_gradient(
#     cmap=cmGreen,
#     subset=[
#         "Difficulty",
#     ],
# )

# c1, c2, c3 = st.columns([1, 3, 1])

# format_dictionary = {
#     "Difficulty": "{:.1%}",
# }

# df = df.format(format_dictionary)

#with c2:
#    st.table(df)


