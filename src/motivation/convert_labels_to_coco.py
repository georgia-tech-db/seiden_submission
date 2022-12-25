



"""

In this file, we write a script that converts from pytorch style to coco file format

"""
from tqdm import tqdm
from utils.file_utils import load_json, save_json
from utils.file_utils import load_torch

def convert(load_file, save_file, classes:list):
    src = load_torch(load_file)

    print(src['categories'])

    result = []
    images = []
    formatted_classes = []

    for idx, cls in enumerate(classes):
        if cls not in src['categories']:
            raise ValueError
        formatted_classes.append({'id': idx, 'name': cls})

    ### now we generate the annotations

    for idx in tqdm(range(len(src['annotations']))):
        anno = src['annotations'][idx]
        for j, label_idx in enumerate(anno['labels']):

            data = {}
            im_data = {}
            im_data['id'] = len(result)
            im_data['height'] = 540
            im_data['width'] = 960
            data['id'] = len(result)
            data['image_id'] = idx
            if int(label_idx) >= len(src['categories']):
                print('asdfasdfasdf', label_idx)
            label = src['categories'][int(label_idx)]
            if label not in classes: continue
            data['category_id'] = classes.index(label)
            x1 = anno['boxes'][j][0].item()
            y1 = anno['boxes'][j][1].item()
            x2 = anno['boxes'][j][2].item()
            y2 = anno['boxes'][j][3].item()
            assert(x2 - x1 >= 0)
            assert(y2 - y1 >= 0)
            data['bbox'] = [x1, y1, x2 - x1, y2 - y1]
            data['score'] = anno['scores'][j].item()
            data['iscrowd'] = 0
            data['area'] = (x2 - x1) * (y2 - y1)
            result.append( data )
            images.append( im_data )

    print(result[0])


    full_result = {'annotations': result, 'images': images, 'categories': formatted_classes}
    save_json(full_result, save_file)
    return result
