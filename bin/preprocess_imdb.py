import argparse as argparse
import csv
import os
import random
from datetime import datetime

import cv2
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

headers = ['filename', 'age', 'gender']


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def load_db(mat_path):
    db = loadmat(mat_path)['imdb'][0, 0]
    num_records = len(db["face_score"][0])

    return db, num_records


def get_meta(db):
    full_path = db["full_path"][0]
    dob = db["dob"][0]  # Matlab serial date number
    gender = db["gender"][0]
    photo_taken = db["photo_taken"][0]  # year
    face_score = db["face_score"][0]
    second_face_score = db["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def main(input_db, photo_dir, output_dir, min_score=1.0, img_size=165, split_ratio=0.8):
    """
    Takes imdb dataset db and performs processing such as cropping and quality checks, writing output to a csv.
    
    :param split_ratio: 
    :param input_db: Path to imdb db
    :param photo_dir: Path to photo's directory
    :param output_dir: Directory to write output to
    :param min_score: minimum score to filter face quality, range [0, 1.0]
    :param img_size: size to crop images to

    """
    crop_dir = os.path.join(output_dir, 'crop')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(crop_dir):
        os.makedirs(crop_dir)

    db, num_records = load_db(input_db)

    indices = list(range(num_records))
    random.shuffle(indices)

    train_indices = indices[:int(len(indices) * split_ratio)]
    test_indices = indices[int(len(indices) * split_ratio):]

    train_csv = open(os.path.join(output_dir, 'train.csv'), 'w')
    train_writer = csv.writer(train_csv, delimiter=',', )
    train_writer.writerow(headers)

    val_csv = open(os.path.join(output_dir, 'val.csv'), 'w')
    val_writer = csv.writer(val_csv, delimiter=',')
    val_writer.writerow(headers)

    clean_and_resize(db, photo_dir, train_indices, min_score, img_size, train_writer, crop_dir)

    clean_and_resize(db, photo_dir, test_indices, min_score, img_size, val_writer, crop_dir)


def clean_and_resize(db, photo_dir, indices, min_score, img_size, writer, crop_dir):
    """
    Cleans records and writes output to :param writer
    :param db:
    :param photo_dir:
    :param indices:
    :param min_score:
    :param img_size: 
    :param crop_dir:
    :param writer: 
    :return: 
    """
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(db)
    for i in tqdm(indices):
        filename = str(full_path[i][0])
        if not os.path.exists(os.path.join(crop_dir, os.path.dirname(filename))):
            os.makedirs(os.path.join(crop_dir, os.path.dirname(filename)))

        img_path = os.path.join(photo_dir, filename)

        if float(face_score[i]) < min_score:
            continue

        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue

        if ~(0 <= age[i] <= 100):
            continue

        if np.isnan(gender[i]):
            continue

        img_gender = int(gender[i])
        img_age = int(age[i])

        img = cv2.imread(img_path)
        crop = cv2.resize(img, (img_size, img_size))
        crop_filepath = os.path.join(crop_dir, filename)
        cv2.imwrite(crop_filepath, crop)

        writer.writerow([filename, img_age, img_gender])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-path', required=True)
    parser.add_argument('--photo-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--min-score', required=False, type=float, default=1.0)
    parser.add_argument('--img-size', type=int, required=False, default=224)
    parser.add_argument('--split-ratio', type=float, required=False, default=0.8)

    args = parser.parse_args()

    main(input_db=args.db_path, photo_dir=args.photo_dir, output_dir=args.output_dir,
         min_score=args.min_score, img_size=args.img_size)
