import tensorflow as tf

def _get_model(model_class, outs=5):
    model_body = model_class(
        include_top=False, weights=None, input_shape=(224, 224, 3))
    model_body.trainable = False

    x = model_body.output
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(outs, activation="softmax")(x)
    return tf.keras.Model(inputs=model_body.input, outputs=x, name=model_body.name)


model_classes = [tf.keras.applications.ResNet50, tf.keras.applications.ResNet50V2,
                 tf.keras.applications.EfficientNetB2, tf.keras.applications.EfficientNetB1, tf.keras.applications.InceptionV3]
models = [_get_model(model_class) for model_class in model_classes]

ResNet50 = models[0]
ResNet50V2 = models[1]
EfficientNetB1 = models[2]
EfficientNetB2 = models[3]
InceptionV3 = models[4]


model_to_func = {
    "ResNet50V1": ResNet50,
    "ResNet50V2": ResNet50V2,
    "EfficientNetB1": EfficientNetB1,
    "EfficientNetB2": EfficientNetB2,
    "InceptionV3": InceptionV3
}