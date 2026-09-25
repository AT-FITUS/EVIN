import os
import gzip
import json
import re
from tqdm import tqdm
from utils import gzip_lines, gzip_objects


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_rating_path", type=str, default="data/amazon_2023/benchmark/5core/rating_only/Musical_Instruments.csv.gz")
    parser.add_argument("-m", "--meta_path", type=str, default="data/amazon_2023/meta_Musical_Instruments.jsonl.gz")
    parser.add_argument("-o", "--output_dir", type=str, default="data/amazon_2023/musical")
    args = parser.parse_args()
    return args


args = parse_arguments()

keys = set()

for l in tqdm(gzip_lines(args.input_rating_path), desc="Loading keys"):
    keys.add(l.split(',')[1])

item_text_f = open(os.path.join(args.output_dir, "item_text.txt"), 'w')
item_image_f = open(os.path.join(args.output_dir, "item_image.txt"), 'w')
for o in tqdm(gzip_objects(args.meta_path), "Exporting metadata"):
    key = o['parent_asin']
    if key in keys:
        text = o['title'] + ' '.join(o['description']) + ' '.join('{} {}'.format(k, v) for k, v in o['details'].items())
        text = re.sub(r'[\t\n\r]+', ' ', text)
        item_text_f.write("{}\t{}\n".format(key, text))
        if len(o["images"]) > 0:
            item_image_f.write("{}\t{}\n".format(key, "\t".join(image['large'] for image in o['images'])))

item_text_f.close()
item_image_f.close()