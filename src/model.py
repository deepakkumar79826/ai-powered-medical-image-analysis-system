import tensorflow as tf


def build_model(img_size=(224, 224), learning_rate=1e-4):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name='data_augmentation',
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_size + (3,),
        include_top=False,
        weights='imagenet',
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=img_size + (3,), name='input_image')
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='prediction')(x)

    model = tf.keras.Model(inputs, outputs, name='medical_image_classifier')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ],
    )
    return model
