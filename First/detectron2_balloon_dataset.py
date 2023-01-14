import os
import json
import cv2
import numpy as np
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
import random
from detectron2.data import MetadataCatalog, DatasetCatalog

def balloon_dataset(image_dir):
    json_file = os.path.join(image_dir, 'via_region_data.json')
    with open(json_file) as f:
        annotations = json.load(f)
    
    # 모든 이미지들의 정보를 넣기 위한 배열을 선언합니다.
    dataset_array = []
    
    for index, value in enumerate(annotations.values()):
        
        # Detectron2의 Dataset 형태는 다음과 같습니다. List[Dictionary]
        dictionary = {}
        
        # 이미지 경로 변수 선언 및 저장하며 이미지 크기까지 선언합니다.
        filename = os.path.join(image_dir, value['filename'])
        height, width = cv2.imread(filename).shape[:2]
        
        # Dictionary에 크기 이름 파일 id를 저장합니다.
        dictionary['file_name'] = filename
        dictionary['image_id'] = index
        dictionary['height'] = height
        dictionary['width'] = width
        
        # 바운딩 박스의 정보를 저장하기 위해 Region 키값을 통해서 불러옵니다.
        annotation = value['regions']
        objects = []
        
        for _, anno in annotation.items():
            ann = anno['shape_attributes']
            # 바운딩 박스 좌표들과 폴리곤 정보를 받아옵니다.
            point_x = ann['all_points_x']
            point_y = ann['all_points_y']
            polygon = [(x + 0.5, y + 0.5) for x, y in zip(point_x, point_y)]
            polygon = [p for x in polygon for p in x]
            
            # 각 객체에 해당하는 바운딩 박스 좌표, 바운딩 박스 설정, 분할 정보, category id를 설정합니다.
            # 이 데이터셋에 대해서는 풍선밖에 없으므로 고정적으로 0으로 설정하며
            # Detectron2의 바운딩 박스 모드 설정을 위해 라이브러리를 추가하여 모드를 설정합니다.
            obj = {
                "bbox" : [np.min(point_x), np.min(point_y), np.max(point_x), np.max(point_y)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [polygon],
                "category_id": 0,
            }
            
            objects.append(obj)
        dictionary['annotations'] = objects
        dataset_array.append(dictionary)
    
    return dataset_array

for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: balloon_dataset("balloon/" + d))
