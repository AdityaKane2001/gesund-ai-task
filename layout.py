import streamlit as st
import runner
import utils
import pandas as pd
import matplotlib.pyplot as plt
import os

class DisplayManager:
    def __init__(self):
        st.set_page_config(page_title="predikt", layout="wide")
        self.model_manip = runner.ModelHandler()
        st.write("*Dev version*")
        self.init_elements()
        self.display_menus()
        self.declare_functionallity()

    def init_elements(self):
        # Input elements
        self.input_container = st.container()
        self.upload_container = st.container()

        # Output elements
        self.output_container = st.container()
        self.images_container = st.container()

    def display_menus(self):
        self.uploaded_weights = None
        self.uploaded_images = None
        self.uploaded_labels = None
        self.uploaded_args_to_names = None

        with self.input_container:
            model_col, mode_col = st.columns(2)
            with model_col:
                model_family = st.selectbox(
                    "Model", ["ResNet50", "EfficientNet", "InceptionV3"])
                if model_family == "ResNet50":
                    model_type = st.selectbox("Variant", ["V1", "V2"])


                elif model_family == "EfficientNet":
                    model_type = st.selectbox("Variant", ["B1", "B2"])
                elif model_family == "InceptionV3":
                    model_type = ""
                self.final_model = model_family + model_type
            with mode_col:
                self.mode = st.radio("Predict or evaluate?",
                                ["Evaluate", "Predict"])
        
        with self.upload_container:
            upload_cols = st.columns(4)
            with upload_cols[0]:
                self.uploaded_weights = st.file_uploader(
                    "Upload model weights here.")
            with upload_cols[1]:
                self.uploaded_args_to_names = st.file_uploader(
                    "Upload integer label to names mapping file here.")
            with upload_cols[2]:
                self.uploaded_images     = st.file_uploader(
                    "Upload images here.", accept_multiple_files=True)

            if self.mode == "Evaluate":
                with upload_cols[3]:
                    self.uploaded_labels = st.file_uploader(
                        "Upload labels file here.")
        
        with self.output_container:
            self.run_button = st.button("Run!")
            self.reset_button = st.button("Reset experiment")
            self.plots, self.metrics = st.columns(2)
    
        
        if self.uploaded_weights is not None:
            self.model_path = utils.write_to_file(self.uploaded_weights)

    def declare_functionallity(self):
        if self.run_button:
            metrics, predictions = self.model_manip.run(self.get_artifacts())
            
            if predictions is None:
                st.error(metrics) # Error object
            else:
                classwise_metrics = self.model_manip.get_classwise_stats()
                self.display_predictions(classwise_metrics)
                self.display_metrics(predictions, metrics)
                
                # self.display_expandable_images(image_label_pairs)
            
            
        
        if self.reset_button:
            self.clear_outputs()


    def display_predictions(self, classwise_stats):
        fig = plt.figure(figsize=(10, 5))
        plt.bar(classwise_stats["Classes"], classwise_stats["Occurences"], color='maroon',width=0.4)
        with self.plots:
            st.pyplot(fig)

    
    def display_metrics(self, predictions, metrics):
        with self.metrics:
            if metrics!="NA":    
                st.write("Average accuracy: **" + str(metrics[1]*100) + "%**")
                st.write("Average loss: **" + str(metrics[0]) + "**")
            df = pd.DataFrame(predictions, columns=[
                              "Image Names", "Raw labels", "Confidence", "Label"])
            # predictions.insert(0, )
            st.table(data=df)
    
    def clear_outputs(self):
        self.output_container.empty() # Works because container may be a subclass of st.empty

    def get_artifacts(self):
        all_assets = [self.final_model, self.mode, self.model_path, self.uploaded_args_to_names,
                      self.uploaded_images, self.uploaded_labels]

        
        artifacts = utils.Artifacts(*all_assets)
                    
        if self.mode == "Evaluate":
            if None in all_assets:
                self.output_container.empty()
                self.output_container.write(
                    "All required artifacts must be provided.")
                raise ValueError("All required artifacts must be provided.")
            
            return artifacts
        else:
            if None in all_assets[:-1]:
                self.output_container.empty()
                self.output_container.write(
                    "All required artifacts must be provided.")
                raise ValueError("All required artifacts must be provided.")
            return artifacts

    
