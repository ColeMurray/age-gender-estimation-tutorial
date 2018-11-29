import argparse

import tensorflow as tf

from age_gender_estimation_tutorial.cnn_estimator import model_fn, serving_fn
from age_gender_estimation_tutorial.dataset import csv_record_input_fn

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--img-dir')
    parser.add_argument('--train-csv')
    parser.add_argument('--val-csv')
    parser.add_argument('--model-dir')
    parser.add_argument('--img-size', type=int, default=160)
    parser.add_argument('--num-steps', type=int, default=200000)

    args = parser.parse_args()

    config = tf.estimator.RunConfig(model_dir=args.model_dir,
                                    save_checkpoints_steps=1500,

                                    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, config=config, params={
            'learning_rate': 0.0001
        })

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: csv_record_input_fn(args.img_dir, args.train_csv, args.img_size, shuffle=False),
        max_steps=args.num_steps,
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: csv_record_input_fn(args.img_dir, args.val_csv, args.img_size, batch_size=1, shuffle=False,
                                             random=False),
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    estimator.export_savedmodel(export_dir_base='{}/serving'.format(args.model_dir),
                                serving_input_receiver_fn=serving_fn,
                                as_text=True)
