import os, sys
import glob

import pandas as pd
import cv2
from joblib import Parallel, delayed


bb_data = pd.read_csv("train_test_bb_fastai_09.csv")
input_folder= sys.argv[1]
output_folder= sys.argv[2]

all_images = glob.glob(f"{input_folder}/*")

def crop_image(path):
    basename = os.path.basename(path)
    bb       = bb_data[bb_data.paths == basename]
    image    = cv2.imread(path)
    H, W, C = image.shape
    x0,y0,x1,y1 = int(bb.x0.values[0] * W), int(bb.y0.values[0] * H), int(bb.x1.values[0] * W), int(bb.y1.values[0] * H)
    image_croped = image[y0 : y1 , x0 : x1]
    cv2.imwrite(f"{output_folder}/{basename}", image_croped)
    

os.makedirs(output_folder, exist_ok = True)
Parallel(n_jobs=2) (delayed(crop_image) (image) for image in all_images)
