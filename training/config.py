import os
from dataclasses import dataclass

@dataclass
class TrainConfig():
    exp_name: str = 'test'
    root_dir: str = r'C:\Users\shiri\Documents\School\Galit\Data\VGG-Face2\data'
    num_classes: int = 10
    train_num_instances_per_class: int =50
    val_num_instances_per_class: int = 30

    epochs:int = 10
    batch_size: int = 8
