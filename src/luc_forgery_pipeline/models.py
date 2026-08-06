from __future__ import annotations

from typing import Callable, Dict

import tensorflow as tf
from tensorflow.keras import layers

from .config import validate_model_name
from .losses_metrics import bce_dice_loss, dice_coefficient, iou_coefficient


def conv_block(x, filters: int, name: str):
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)
    x = layers.Conv2D(filters, 3, padding="same", kernel_initializer="he_normal", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def decoder_block(x, skip, filters: int, name: str):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding="same", name=f"{name}_up")(x)
    if skip is not None:
        x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    return conv_block(x, filters, name=name)


def _last_feature_at_resolution(base: tf.keras.Model, resolution: int):
    """Return the last 4D feature tensor with the requested spatial resolution."""
    candidates = []
    for layer in base.layers:
        output = getattr(layer, "output", None)
        shape = getattr(output, "shape", None)
        if shape is None or len(shape) != 4:
            continue
        height, width = shape[1], shape[2]
        if height == resolution and width == resolution:
            candidates.append(output)
    if not candidates:
        raise ValueError(f"No feature map with resolution {resolution}x{resolution} found in {base.name}")
    return candidates[-1]


def build_plain_unet(img_size: int) -> tf.keras.Model:
    inputs = layers.Input((img_size, img_size, 3), name="image")
    c1 = conv_block(inputs, 32, "enc1")
    p1 = layers.MaxPooling2D(name="pool1")(c1)
    c2 = conv_block(p1, 64, "enc2")
    p2 = layers.MaxPooling2D(name="pool2")(c2)
    c3 = conv_block(p2, 128, "enc3")
    p3 = layers.MaxPooling2D(name="pool3")(c3)
    c4 = conv_block(p3, 256, "enc4")
    p4 = layers.MaxPooling2D(name="pool4")(c4)
    bridge = conv_block(p4, 512, "bridge")

    d4 = decoder_block(bridge, c4, 256, "dec4")
    d3 = decoder_block(d4, c3, 128, "dec3")
    d2 = decoder_block(d3, c2, 64, "dec2")
    d1 = decoder_block(d2, c1, 32, "dec1")
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="mask")(d1)
    return tf.keras.Model(inputs, outputs, name="plain_unet")


def build_unetplusplus(img_size: int) -> tf.keras.Model:
    inputs = layers.Input((img_size, img_size, 3), name="image")
    f = [32, 64, 128, 256, 512]

    x00 = conv_block(inputs, f[0], "x00")
    x10 = conv_block(layers.MaxPooling2D(name="pool10")(x00), f[1], "x10")
    x20 = conv_block(layers.MaxPooling2D(name="pool20")(x10), f[2], "x20")
    x30 = conv_block(layers.MaxPooling2D(name="pool30")(x20), f[3], "x30")
    x40 = conv_block(layers.MaxPooling2D(name="pool40")(x30), f[4], "x40")

    x01 = conv_block(layers.Concatenate(name="cat01")([x00, layers.UpSampling2D(name="up01")(x10)]), f[0], "x01")
    x11 = conv_block(layers.Concatenate(name="cat11")([x10, layers.UpSampling2D(name="up11")(x20)]), f[1], "x11")
    x21 = conv_block(layers.Concatenate(name="cat21")([x20, layers.UpSampling2D(name="up21")(x30)]), f[2], "x21")
    x31 = conv_block(layers.Concatenate(name="cat31")([x30, layers.UpSampling2D(name="up31")(x40)]), f[3], "x31")

    x02 = conv_block(layers.Concatenate(name="cat02")([x00, x01, layers.UpSampling2D(name="up02")(x11)]), f[0], "x02")
    x12 = conv_block(layers.Concatenate(name="cat12")([x10, x11, layers.UpSampling2D(name="up12")(x21)]), f[1], "x12")
    x22 = conv_block(layers.Concatenate(name="cat22")([x20, x21, layers.UpSampling2D(name="up22")(x31)]), f[2], "x22")

    x03 = conv_block(layers.Concatenate(name="cat03")([x00, x01, x02, layers.UpSampling2D(name="up03")(x12)]), f[0], "x03")
    x13 = conv_block(layers.Concatenate(name="cat13")([x10, x11, x12, layers.UpSampling2D(name="up13")(x22)]), f[1], "x13")

    x04 = conv_block(
        layers.Concatenate(name="cat04")([x00, x01, x02, x03, layers.UpSampling2D(name="up04")(x13)]),
        f[0],
        "x04",
    )
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="mask")(x04)
    return tf.keras.Model(inputs, outputs, name="unetplusplus")


def build_efficientnetb0_unet(img_size: int, weights: str | None = "imagenet") -> tf.keras.Model:
    try:
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=weights,
            input_shape=(img_size, img_size, 3),
        )
    except Exception as exc:
        print(f"EfficientNetB0 ImageNet weights unavailable ({exc}); falling back to random initialization.")
        base = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=(img_size, img_size, 3),
        )
    x = base.output
    bottleneck_resolution = int(x.shape[1])
    skip_resolutions = [bottleneck_resolution * (2**i) for i in range(1, 5)]
    skips = [_last_feature_at_resolution(base, resolution) for resolution in skip_resolutions]
    for i, (skip, filters) in enumerate(zip(skips, [256, 128, 64, 32]), start=1):
        x = decoder_block(x, skip, filters, f"dec{i}")
    x = decoder_block(x, None, 16, "dec_final")
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="mask")(x)
    return tf.keras.Model(base.input, outputs, name="efficientnetb0_unet")


def build_resnet50_unet(img_size: int, weights: str | None = "imagenet") -> tf.keras.Model:
    try:
        base = tf.keras.applications.ResNet50(
            include_top=False,
            weights=weights,
            input_shape=(img_size, img_size, 3),
        )
    except Exception as exc:
        print(f"ResNet50 ImageNet weights unavailable ({exc}); falling back to random initialization.")
        base = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_shape=(img_size, img_size, 3),
        )
    skips = [
        base.get_layer("conv4_block6_out").output,
        base.get_layer("conv3_block4_out").output,
        base.get_layer("conv2_block3_out").output,
        base.get_layer("conv1_relu").output,
    ]
    x = base.get_layer("conv5_block3_out").output
    for i, (skip, filters) in enumerate(zip(skips, [256, 128, 64, 32]), start=1):
        x = decoder_block(x, skip, filters, f"dec{i}")
    x = decoder_block(x, None, 16, "dec_final")
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="mask")(x)
    return tf.keras.Model(base.input, outputs, name="resnet50_unet")


BUILDERS: Dict[str, Callable[[int], tf.keras.Model]] = {
    "plain_unet": build_plain_unet,
    "unetplusplus": build_unetplusplus,
    "efficientnetb0_unet": build_efficientnetb0_unet,
    "resnet50_unet": build_resnet50_unet,
}


def build_model(model_name: str, img_size: int) -> tf.keras.Model:
    validate_model_name(model_name)
    return BUILDERS[model_name](img_size)


def compile_model(model: tf.keras.Model, learning_rate: float, optimizer_name: str = "adamw") -> tf.keras.Model:
    optimizer_key = optimizer_name.lower()
    if optimizer_key == "adamw" and hasattr(tf.keras.optimizers, "AdamW"):
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=bce_dice_loss,
        metrics=[
            dice_coefficient,
            iou_coefficient,
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5),
            tf.keras.metrics.Precision(name="precision", thresholds=0.5),
            tf.keras.metrics.Recall(name="recall", thresholds=0.5),
        ],
    )
    return model
