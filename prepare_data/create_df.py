import os
import pandas as pd
from tqdm import tqdm
from face_detection.training.config import TrainConfig


def create_unique_df(config):
    data_dir = os.path.join(config.root_dir, 'train')
    df = pd.DataFrame(columns=[['class', 'instance_num', 'image_name', 'image_path','set']])
    classes = os.listdir(data_dir)
    for class_num,c in tqdm(enumerate(classes[0:config.num_classes+1])):
        class_path = os.path.join(data_dir, c)
        imgs = os.listdir(class_path)
        for t,img in enumerate(imgs[0:config.train_num_instances_per_class+1]):
            img_path = os.path.join(class_path, img)
            row = [class_num, t, img, img_path,'train']
            df.loc[len(df)]=row
        for v,img in enumerate(imgs[config.train_num_instances_per_class+1: t + config.val_num_instances_per_class+1]):
            img_path = os.path.join(class_path, img)
            row = [class_num, t+v+1, img, img_path,'val']
            df.loc[len(df)]=row

    return df

if __name__ == '__main__':
    config = TrainConfig()
    set = 'train'
    num_classes = getattr(config, f'num_classes')
    num_instances_per_class = getattr(config, f'{set}_num_instances_per_class')
    df_path = f'{config.root_dir}/{config.exp_name}_{num_classes}_{num_instances_per_class}.csv'
    try:
        df = pd.read_csv(df_path)
    except:
        df = create_unique_df(config)

    df.to_csv(df_path)