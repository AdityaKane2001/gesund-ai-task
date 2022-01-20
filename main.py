import streamlit as st
import tensorflow as tf
from streamlit import write

from layout import DisplayManager
# from runner import get_predict_ds, get_eval_ds
# from models import model_to_func
# from utils import write_to_file


st.set_page_config(page_title="predikt", layout="wide")
# st.write("Select model: \n")

# model_to_class = {
#     "ResNet50V1" : ResNet50,
#     "ResNet50V2": ResNet50V2,
#     "EfficientNetB1": EfficientNetB1,
#     "EfficientNetB2": EfficientNetB2,
#     "InceptionV3": InceptionV3
# }

master = DisplayManager()

#------------------- Model selection------------------#

# model_family = st.selectbox("Model", ["ResNet50", "EfficientNet", "InceptionV3"])

# if model_family == "ResNet50":
#     model_type = st.selectbox("Variant",["V1", "V2"])
# elif model_family == "EfficientNet":
#     model_type = st.selectbox("Variant", ["B1","B2"])
# elif model_family == "InceptionV3":
#     model_type = ""
# final_model = model_family + model_type

#-----------------Mode selection---------------------#

# mode = st.radio("Predict or evaluate?", ["Evaluate", "Predict"])
# uploaded_weights = None
# uploaded_images = None
# uploaded_weights = st.file_uploader("Upload model weights here.")
# uploaded_args_to_names = st.file_uploader(
#     "Upload integer label to names mapping file here.")
# uploaded_images = st.file_uploader(
#     "Upload images here.", accept_multiple_files=True)

# if mode=="Evaluate":
#     uploaded_labels = st.file_uploader("Upload labels file here.")

# if uploaded_weights is not None:
#     model_path = write_to_file(uploaded_weights)

# if uploaded_images is not None:
#     batch_size = len(uploaded_images)

#--------------------Eval / Predict-------------------------#

# run_button = st.button("Run!")
# reset_button = st.button("Reset experiment")

# container = st.container()

# if run_button:
#     # st.write("Hello")
    
#     model = model_to_class[final_model]
#     model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
#     if mode == "Evaluate":
#         ds, mappings = get_eval_ds(uploaded_images, uploaded_labels,
#                          uploaded_args_to_names)
#         ds = ds.batch(batch_size)
#         model.load_weights(model_path)
#         metrics = model.evaluate(ds)
#         logits = model.predict(ds)
#         container.write(metrics)
#         container.write(logits)
#     elif mode == "Predict":
#         ds, mappings = get_predict_ds(uploaded_images, uploaded_args_to_names)
#         ds = ds.batch(batch_size)
#         logits = model.predict(ds)


# if reset_button:
#     container.empty()

