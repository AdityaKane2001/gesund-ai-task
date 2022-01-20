from asyncio.proactor_events import _ProactorSocketTransport
from tkinter import image_names
import tensorflow as tf
import numpy as np

from models import model_to_func

class DataHandler:
    def __init__(self):
        pass
    
    def parse_label_mapping(self, args_to_names):
        mappings = args_to_names.getvalue().decode("utf-8").split("\n")
        mappings = [pair.strip("\n") for pair in mappings]
        mappings = [pair.split(" : ") for pair in mappings]
        mappings = [(int(pair[0]), pair[1]) for pair in mappings]
        mappings = {key: value for (key, value) in mappings}
        return mappings
    
    def parse_image(self, imagepath):
        # st.write()
        image_string = imagepath.getvalue()
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, size=[224, 224])
        image = image * 255.
    #     label = label_mapping[tf.strings.split(tf.strings.split(image_path, sep="/")[-1], sep="_")[0]]
        return image, imagepath.name

    def parse_labels(self, labels_file, mapping):
        all_labels = labels_file.getvalue().decode("utf-8").split("\n")
        all_labels = [label.strip("\r").split(" : ") for label in all_labels]
        all_labels = {key: int(mapping[value]) for key, value in all_labels}
        return all_labels

    def get_predict_images(self, image_list):
        return [self.parse_image(imagepath)[0] for imagepath in image_list]

    def get_eval_images(self, image_list):
        return [self.parse_image(imagepath) for imagepath in image_list]


    def get_predict_ds(self, images, args_to_names):
        mappings = self.parse_label_mapping(args_to_names)
        print(mappings)
        all_images = np.array(self.get_predict_images(images))
        all_images = all_images[np.newaxis, :]
        all_images_ds = tf.data.Dataset.from_tensor_slices(all_images)
        return all_images_ds, mappings

    def get_eval_ds(self, images, labels, args_to_names):
        int_to_label_mappings = self.parse_label_mapping(args_to_names)
        label_to_int_mapping = {v: k for k, v in int_to_label_mappings.items()}

        self.int_to_label = int_to_label_mappings
        self.label_to_int = label_to_int_mapping

        print(label_to_int_mapping)
        print(int_to_label_mappings)
        
        images_and_imagenames = self.get_eval_images(images)

        images, imagenames = list(zip(*images_and_imagenames))
        self.imagenames = imagenames

        label_mappings = self.parse_labels(labels, label_to_int_mapping)
        labels = [label_mappings[imagename]
                for imagename in imagenames]

        all_images = np.array(images)
        # all_images = all_images[:]
        all_images_ds = tf.data.Dataset.from_tensor_slices(all_images)

        all_labels_ds = tf.data.Dataset.from_tensor_slices(labels)
        all_labels_ds = all_labels_ds.map(
            lambda x: tf.one_hot(x, depth=5, dtype=tf.float32))

        ds = tf.data.Dataset.zip((all_images_ds, all_labels_ds))
        return ds, int_to_label_mappings

    def get_imagelist(self):
        return self.imagenames

class ModelHandler:
    def __init__(self):
        # We initialize required things when `Run!` button is clicked. 
        self.data_handler = DataHandler()

    def postprocess_logits(self, logits):
        all_imagenames = self.data_handler.get_imagelist()
        all_argmaxes = np.argmax(logits, axis=-1)
        all_confidences = logits.max(axis=-1)
        all_labels = [self.data_handler.int_to_label[agmax] for agmax in all_argmaxes]

        results = list(zip(all_imagenames, all_argmaxes, all_confidences, all_labels))
        return results


    def run(self, artifacts):
        model = model_to_func[artifacts.final_model]
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        if artifacts.mode == "Evaluate":
            ds, mappings = self.data_handler.get_eval_ds(artifacts.uploaded_images,
                                            artifacts.uploaded_labels,
                                            artifacts.uploaded_args_to_names)
            ds = ds.batch(len(artifacts.uploaded_images))
            model.load_weights(artifacts.model_path)
            metrics = model.evaluate(ds)
            logits = model.predict(ds)
            return metrics, self.postprocess_logits(logits)
        elif artifacts.mode == "Predict":
            ds, mappings = self.data_handler.get_predict_ds(
                self.uploaded_images, self.uploaded_args_to_names)
            ds = ds.batch(len(artifacts.uploaded_images))
            logits = model.predict(ds)
            return "NA", self.postprocess_logits(logits)
