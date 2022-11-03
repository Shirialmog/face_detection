import os
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    train_dir = r'C:\Users\shiri\Documents\School\Galit\Data\VGG-Face2\data\train'
    data_csv = pd.DataFrame(columns=[['class','instance_num', 'image_name', 'image_path' ]])
    classes = os.listdir(train_dir)
    for c in tqdm(classes[0:100]):
        class_path = os.path.join(train_dir, c)
        imgs = os.listdir(class_path)
        for i,img in enumerate(imgs):
            img_path = os.path.join(class_path, img)
            row = [c, i, img, img_path]
            data_csv.loc[len(data_csv)]=row

    data_csv.to_csv(r'C:\Users\shiri\Documents\School\Galit\Data\VGG-Face2\data\train_data.csv')