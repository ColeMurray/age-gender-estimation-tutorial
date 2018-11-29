import logging
from argparse import ArgumentParser

import tensorflow as tf
from scipy.misc import imread
from tensorflow.contrib import predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--image-path', required=True)

    args = parser.parse_args()

    prediction_fn = predictor.from_saved_model(export_dir=args.model_dir, signature_def_key='serving_default')

    batch = []

    image = imread(args.image_path)
    output = prediction_fn({
        'image': [image]
    })
    print(output)
