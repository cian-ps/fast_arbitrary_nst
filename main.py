import tensorflow as tf
from pathlib import Path
import os
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--content", dest="content", help="path to content image")
parser.add_argument("--style", dest="style", help="path to style image")
args = parser.parse_args()

IMAGE_DIR = Path(os.path.dirname(os.path.realpath(__file__)))/"generated_images"
IMAGE_DIR.mkdir(exist_ok=True)

def im_loader(path):
    im = tf.image.decode_image(tf.io.read_file(path),
                                channels=3,
                                dtype=tf.float32)
    original_shape = im.shape[:2]
    im = tf.image.resize(im, (512, 512))
    return (tf.expand_dims(im, axis=0), original_shape)

def save_and_display(tensor):
    filename = str(Path(args.style).stem) + "_" + str(Path(args.content).name)
    im = tf.keras.utils.array_to_img(tensor)
    print(f"saving image to {str(IMAGE_DIR/filename)}")
    im.save(str(IMAGE_DIR/filename))
    Image.open(str(IMAGE_DIR/filename)).show()

def load_pb():
    model_dir = Path(os.path.dirname(os.path.realpath(__file__)))/"model"
    return tf.saved_model.load(model_dir)

with tf.device("/CPU:0"):
    print(f"loading image: {args.content}")
    content, original_shape = im_loader(args.content)
    print(f"loading image: {args.style}")
    style, _ = im_loader(args.style)

    model = load_pb()

    print("extracting features...")
    output = model(content, style)
    output = tf.image.resize(tf.squeeze(output[0], axis=0), original_shape)
    save_and_display(output)