import streamlit as st
import tensorflow as tf
from streamlit import write

from runner import get_predict_ds, get_eval_ds
from models import ResNet50, ResNet50V2, EfficientNetB1, EfficientNetB2, InceptionV3

st.set_page_config(page_title="predikt")

st.write("Select model: \n")

model_to_class = {
    "ResNet50V1" : ResNet50,
    "ResNet50V2": ResNet50V2,
    "EfficientNetB1": EfficientNetB1,
    "EfficientNetB2": EfficientNetB2,
    "InceptionV3": InceptionV3
}

#------------------- Model selection------------------#

model_family = st.selectbox("Model", ["ResNet50", "EfficientNet", "InceptionV3"])

if model_family == "ResNet50":
    model_type = st.selectbox("Variant",["V1", "V2"])
    final_model = model_family + model_type
elif model_family == "EfficientNet":
    model_type = st.selectbox("Variant", ["B1","B2"])
    final_model = model_family + model_type
elif model_family == "InceptionV3":
    model_type = None
    final_model = model_family



#-----------------Mode selection---------------------#

mode = st.radio("predict or evaluate?", ["Evaluate", "Predict"])

if mode=="Evaluate":
    uploaded_weights = st.file_uploader("Upload model weights here.")
    uploaded_args_to_names = st.file_uploader("Upload integer label to names mapping file here.")
    uploaded_images = st.file_uploader("Upload images here.", accept_multiple_files=True)
    uploaded_labels = st.file_uploader("Upload labels file here.")
elif mode=="Predict":
    uploaded_weights = st.file_uploader("Upload model weights here.")
    uploaded_args_to_names = st.file_uploader(
        "Upload interger label to names mapping file here.")
    uploaded_images = st.file_uploader(
        "Upload images here.", accept_multiple_files=True)

#--------------------Predict-------------------------#

run_button = st.button("Run!")
reset_button = st.button("Reset experiment")

if run_button:
    # st.write("Hello")
    
    model = model_to_class[final_model]
    model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
    if mode == "Evaluate":
        ds, mappings = get_eval_ds(uploaded_images, uploaded_labels,
                         uploaded_args_to_names)
        for i in ds:
            print(i)
            break
        # metrics = model.evaluate(ds)
    elif mode == "Predict":
        ds, mappings = get_predict_ds(uploaded_images, uploaded_args_to_names)
        for i in ds:
            print(i)
            break
        # logits = model.predict(ds)


if reset_button:
    st.write()
