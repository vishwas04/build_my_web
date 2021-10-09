from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

import numpy as np
import cv2

from tensorflow.keras.layers.experimental.preprocessing import StringLookup
AUTOTUNE = tf.data.AUTOTUNE
max_len = 21
characters=['i', 'K', '/', 'T', 'z', 'a', ')', '+', 'D', 'n', '#', '3', 'd', 'S', '6', 's', 'h', 'p', 'W', ':', 'k', 'I', 'w', 'y', '!', 'H', 'R', 'q', 'o', '5', 'M', 'l', '8', '&', 'Y', 'L', 'P', '?', 'N', 'O', ',', 'v', '-', 'c', ';', 'C', 'G', '1', 'g', 'e', 'J', 'u', 'r', '"', 'j', '0', 'E', 'F', 'x', 'b', 'V', 'm', 'X', '.', 'Z', '7', 't', '*', '2', 'U', 'B', "'", 'Q', '4', 'f', '(', 'A', '9']

# Mapping characters to integers.
char_to_num = StringLookup(vocabulary=characters, mask_token=None)

# Mapping integers back to original characters.
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None,**kwargs):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost
#         super(CustomLayer, self).__init__(name=name)
#         self.k = k
        super(CTCLayer, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred


def distortion_free_resize(image, img_size):
    print(image)
    w, h = img_size
    image  = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2
    
    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
                  [pad_height_top, pad_height_bottom],
                  [pad_width_left, pad_width_right],
                  [0, 0]
                ]
        )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

new_model = tf.keras.models.load_model('model_20.h5', custom_objects={'CTCLayer': CTCLayer})
# print(new_model.weights)
def load_images_from_folder(folder):
    images = []
    file=[]
    ids=[]
    ii=1
    for filename in os.listdir(folder):
        if(str(filename)[0]!='(' or str(filename)[-1]!=')'):
            continue
        # print(filename)
        for img in os.listdir(folder+str(filename)+"/detect/"):
            file.append(folder+str(filename)+"/detect/"+str(img))
            pos=str(img)[:-4].split("_")
            ids.append(filename+"/"+pos[-2]+"_"+pos[-1])

            
#         img = cv2.imread(os.path.join(folder,filename))
#         if img is not None:
# #             images.append(img)
#             file.append(folder+str(filename))
#             ids.append(str(ii))
#             ii=ii+1
   
    return file,ids
f,ids=load_images_from_folder("/Users/vishwas/Desktop/build_my_web/WordDetectorNN-master/src/result/")
# print(f,ids)

batch_size = 64
padding_token = 99
image_width = 128
image_height = 32


def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.
    return image


def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label


def process_images_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels):
    #creats a something like generator and applies a funtion for each element (.map)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
        process_images_labels, num_parallel_calls=AUTOTUNE
    )
    #batch is like grouping into buckets 
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)


v_test=prepare_dataset(f, ids)
prediction_model = keras.models.Model(
    new_model.get_layer(name="image").input, new_model.get_layer(name="dense2").output
)
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_len
    ]
    
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        # print(res)[27 57 57  4 57 78 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        # print(res)[27 57 57  4 57 78]
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
#     with open('/Users/vishwas/Desktop/bd/model.txt', 'w') as f:
#         f.write(str(output_text))
    return output_text


#  Let's check results on some test samples.
for batch in v_test.take(1):
    batch_images = batch["image"]
    # _, ax = plt.subplots(5, 5, figsize=(15, 8))

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    for i in range(len(batch["image"])):
        # img = batch_images[i]
        # img = tf.image.flip_left_right(img)
        # img = tf.transpose(img, perm=[1, 0, 2])
        # img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
        # img = img[:, :, 0]
        # gggg=5
        print(pred_texts[i],ids[i])

        # ax[i // gggg, i % gggg].imshow(img, cmap="gray")
        # ax[i // gggg, i % gggg].set_title(title)
        # ax[i // gggg, i % gggg].axis("off")