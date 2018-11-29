import os

import tensorflow as tf


def csv_record_input_fn(img_dir, filenames, img_size=150, repeat_count=-1, shuffle=True,
                        batch_size=16, random=True):
    """
    Creates tensorflow dataset iterator over records from :param{filenames}.
    
    :param img_dir: Path to directory of cropped images
    :param filenames: array of file paths to load rows from
    :param img_size: size of image
    :param repeat_count: number of times for iterator to repeat
    :param shuffle: flag for shuffling dataset
    :param batch_size: number of examples in batch
    :param random: flag for random distortion to the image
    :return: Iterator of dataset
    """

    def parse_csv_row(line):
        defaults = [[""], [0], [0]]
        filename, age, gender = tf.decode_csv(line, defaults)
        filename = os.path.join(img_dir) + '/' + filename

        image_string = tf.read_file(filename)
        image = tf.image.decode_image(image_string, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.per_image_standardization(image)
        image.set_shape([img_size, img_size, 3])

        age = tf.cast(age, tf.int64)
        gender = tf.cast(gender, tf.int64)

        if random:
            image = tf.image.random_flip_left_right(image)

        return {'image': image}, dict(gender=gender, age=age)

    dataset = tf.data.TextLineDataset(filenames).skip(1)
    dataset = dataset.map(parse_csv_row)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=2000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.prefetch(batch_size * 10)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()
