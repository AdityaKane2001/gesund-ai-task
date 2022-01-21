import tensorflow as tf
import numpy as np

from collections import Counter

from models import model_to_func
from utils import write_to_file

class DataHandler:
    """
    Handles all data-related operations. Any function starting with `parse` 
    reads files and converts contents into relevant python objects.
    
    """
    def __init__(self):
        pass
    
    def check_labels_and_images(self, imagenames, label_mappings):
        sorted_images = imagenames.sort()
        sorted_labels = label_mappings.keys().sort()
        assert sorted_images == sorted_labels, "Check labels and images again"

    def check_labels_and_mappings(self, labels_mapping, label_to_int_mappings):
        for label,_ in labels_mapping:
            assert label in label_to_int_mappings.keys(), "Check labels and label mappings again"

    def parse_label_mapping(self, args_to_names):
        mappings = args_to_names.getvalue().decode("utf-8").split("\n")
        mappings = [pair.strip("\n") for pair in mappings]
        mappings = [pair.split(" : ") for pair in mappings]
        mappings = [(int(pair[0]), pair[1]) for pair in mappings]
        mappings = {key: value for (key, value) in mappings}
        return mappings
    
    def parse_image(self, imagepath):
        write_to_file(imagepath)
        image_string = imagepath.getvalue()
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, size=[224, 224])
        image = image * 255.
        return image, imagepath.name

    def parse_labels(self, labels_file, mapping):
        all_labels = labels_file.getvalue().decode("utf-8").split("\n")
        all_labels = [label.strip("\r").split(" : ") for label in all_labels]

        self.check_labels_and_mappings(all_labels, mapping)

        all_labels = {key: int(mapping[value]) for key, value in all_labels}
        return all_labels

    def get_images_and_names(self, image_list):
        return [self.parse_image(imagepath) for imagepath in image_list]


    def get_predict_ds(self, images, args_to_names):
        int_to_label_mappings = self.parse_label_mapping(args_to_names)
        label_to_int_mapping = {v: k for k, v in int_to_label_mappings.items()}

        images_and_imagenames = self.get_images_and_names(images)
        images, imagenames = list(zip(*images_and_imagenames))

        # all_images = np.array(self.get_predict_images(images))

        all_images = np.array(images)
        # all_images = all_images[:]
        all_images_ds = tf.data.Dataset.from_tensor_slices(all_images)

        self.imagenames = imagenames
        self.int_to_label = int_to_label_mappings
        self.label_to_int = label_to_int_mapping

        return all_images_ds

    def get_eval_ds(self, images, labels, args_to_names):
        int_to_label_mappings = self.parse_label_mapping(args_to_names)
        label_to_int_mapping = {v: k for k, v in int_to_label_mappings.items()}

        
        images_and_imagenames = self.get_images_and_names(images)
        images, imagenames = list(zip(*images_and_imagenames))
        

        label_mappings = self.parse_labels(labels, label_to_int_mapping)
        labels = [label_mappings[imagename]
                for imagename in imagenames]

        self.check_labels_and_images(imagenames, label_mappings)

        all_images = np.array(images)
        # all_images = all_images[:]
        all_images_ds = tf.data.Dataset.from_tensor_slices(all_images)

        all_labels_ds = tf.data.Dataset.from_tensor_slices(labels)
        all_labels_ds = all_labels_ds.map(
            lambda x: tf.one_hot(x, depth=5, dtype=tf.float32))

        ds = tf.data.Dataset.zip((all_images_ds, all_labels_ds))

        # cache important stuff
        self.imagenames = imagenames
        self.int_to_label = int_to_label_mappings
        self.label_to_int = label_to_int_mapping
        return ds

    def get_imagelist(self):
        return self.imagenames

class ModelHandler:
    """Handles all model related operations."""
    def __init__(self):
        self.data_handler = DataHandler()

    def postprocess_logits(self, logits):
        all_imagenames = self.data_handler.get_imagelist()
        all_argmaxes = np.argmax(logits, axis=-1)
        all_confidences = logits.max(axis=-1)
        all_labels = [self.data_handler.int_to_label[agmax] for agmax in all_argmaxes]
        
        self.labels = all_labels # For classwise

        results = list(zip(all_imagenames, all_argmaxes, all_confidences, all_labels))
        
        return results
    
    def get_classwise_stats(self):
        c = Counter(self.labels)
        cols, occ = list(zip(*c.items()))
        return {"Classes": cols, "Occurences": occ}

    def run(self, artifacts):

        model = model_to_func[artifacts.final_model]
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        
        if artifacts.mode == "Evaluate":
            ds = self.data_handler.get_eval_ds(artifacts.uploaded_images,
                                            artifacts.uploaded_labels,
                                            artifacts.uploaded_args_to_names)
            ds = ds.batch(len(artifacts.uploaded_images))

            model.load_weights(artifacts.model_path)
            metrics = model.evaluate(ds)
            logits = model.predict(ds)

            return metrics, self.postprocess_logits(logits)

        elif artifacts.mode == "Predict":
            ds = self.data_handler.get_predict_ds(
                artifacts.uploaded_images, artifacts.uploaded_args_to_names)
            ds = ds.batch(len(artifacts.uploaded_images))

            model.load_weights(artifacts.model_path)
            logits = model.predict(ds)

            return "NA", self.postprocess_logits(logits)
