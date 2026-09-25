import os
import re
from tqdm import tqdm
from utils import gzip_lines, gzip_objects


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rating_path", type=str, default="data/amazon_2023/benchmark/5core/rating_only/Musical_Instruments.csv.gz")
    parser.add_argument("-r", "--review_path", type=str, default="data/amazon_2023/Musical_Instruments.jsonl.gz")
    parser.add_argument("-o", "--output_dir", type=str, default="data/amazon_2023/musical")
    args = parser.parse_args()
    return args


args = parse_arguments()

os.makedirs(args.output_dir, exist_ok=True)

keys = set()

rating_f = open(os.path.join(args.output_dir, "rating.txt"), "w")
for c, l in tqdm(enumerate(gzip_lines(args.input_rating_path)), desc="Loading keys"):
    if c > 0:
        keys.add(l)
        rating_f.write("{}\n".format(l))

rating_f.close()

review_f = open(os.path.join(args.output_dir, "review.txt"), 'w')
review_image_f = open(os.path.join(args.output_dir, "review_image.txt"), 'w')
for o in tqdm(gzip_objects(args.review_path), "Exporting reviews"):
    key = '{},{},{},{}'.format(o['user_id'], o['parent_asin'], o['rating'], o['timestamp'])
    if key in keys:
        if o['text']:
            review = re.sub(r'[\t\n\r]+', ' ', o['text']).strip()
            if len(review) > 0:
                review_f.write("{}\t{}\t{}\n".format(o['user_id'], o['parent_asin'], review))
        if len(o["images"]) > 0:
            review_image_f.write("{}\t{}\t{}\n".format(o['user_id'], o['parent_asin'], '\t'.join(image['small_image_url'] for image in o['images'])))


review_f.close()
review_image_f.close()