import streamlit as st
import numpy as np
from pandas import DataFrame
from flair.embeddings import TransformerDocumentEmbeddings
import seaborn as sns
from functionforDownloadButtons import download_button
import onnxruntime as rt
from fen_to_array import *
import pickle

#sess = rt.InferenceSession("model/chess-resnet.onnx")

sess = rt.InferenceSession("model/chess-resnet.onnx")
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name


scaler = pickle.load(open("model/mm_scaler_untilmove80_clipped500.pkl", "rb"))
def rcnn_chess(fenstring='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'): 
    sample = np.array(fenToVec(fenstring))
    print(sample)
    scores = sess.run(None, {input_name: sample})[0][0]
    print(scores)
    scores = np.array(scores).reshape(1,-1)
    prediction = scaler.inverse_transform(scores)
    print(prediction)
    prediction = np.where(prediction[0][0] > 0, prediction[0][0].item(), 0)
    print(prediction)
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
-   The *BERT Keyword Extractor* app is an easy-to-use interface built in Streamlit for the amazing [KeyBERT](https://github.com/MaartenGr/KeyBERT) library from Maarten Grootendorst!
-   It uses a minimal keyword extraction technique that leverages multiple NLP embeddings and relies on [Transformers] (https://huggingface.co/transformers/) 🤗 to create keywords/keyphrases that are most similar to a document.
	    """
    )

    st.markdown("")

st.markdown("")
st.markdown("## **📌 Paste document **")
with st.form(key="my_form"):


    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    with c1:
        pass


        







    with c2:
        doc = st.text_area(
            "Paste your text below (max 500 words)",
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

        submit_button = st.form_submit_button(label="✨ Get me the data!")



try:
    keywords = rcnn_chess(doc)
except:
    keywords = "Error"

st.markdown("## **🎈 Check & download results **")

st.write(str(keywords))

st.header("")

cs, c1, c2, c3, cLast = st.columns([2, 1.5, 1.5, 1.5, 2])

with c1:
    CSVButton2 = download_button(keywords, "Data.csv", "📥 Download (.csv)")
with c2:
    CSVButton2 = download_button(keywords, "Data.txt", "📥 Download (.txt)")
with c3:
    CSVButton2 = download_button(keywords, "Data.json", "📥 Download (.json)")

st.header("")

df = (
    DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
    .sort_values(by="Relevancy", ascending=False)
    .reset_index(drop=True)
)

df.index += 1

# Add styling
cmGreen = sns.light_palette("green", as_cmap=True)
cmRed = sns.light_palette("red", as_cmap=True)
df = df.style.background_gradient(
    cmap=cmGreen,
    subset=[
        "Relevancy",
    ],
)

c1, c2, c3 = st.columns([1, 3, 1])

format_dictionary = {
    "Relevancy": "{:.1%}",
}

df = df.format(format_dictionary)

with c2:
    st.table(df)
