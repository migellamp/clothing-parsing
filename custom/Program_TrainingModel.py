
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw


ROOT_DIR = os.path.abspath("../../")


sys.path.append(ROOT_DIR)  
from mrcnn.config import Config
from mrcnn import model as modellib, utils


COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Konfigurasi
############################################################


class CustomConfig(Config):
    NAME = "custom"
    IMAGES_PER_GPU = 4
    NUM_CLASSES = 1 + 13 
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        # Inisiasi Class 
        self.add_class("custom", 1, "bag")
        self.add_class("custom", 2, "belt")
        self.add_class("custom", 3, "boots")
        self.add_class("custom", 4, "footwear")
        self.add_class("custom", 5, "outer")
        self.add_class("custom", 6, "dress")
        self.add_class("custom", 7, "sunglasses")
        self.add_class("custom", 8, "pants")
        self.add_class("custom", 9, "top")
        self.add_class("custom", 10, "shorts")
        self.add_class("custom", 11, "skirt")
        self.add_class("custom", 12, "headwear")
        self.add_class("custom", 13, "scarf/tie")

        # Check Folder Path
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        
        # Buka File Anotasi
        annotations = json.load(open(os.path.join(dataset_dir, "annotations2.json")))
        annotations = list(annotations.values())  # don't need the dict keys
        annotations = [a for a in annotations if a['regions']]

        # Baca Isi File Anotasi
        for a in annotations:
            # Ambil Koordinat yang membentuk polygon pada format shape_attributes
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            custom = [s['region_attributes'] for s in a['regions'].values()]
            
            num_ids=[]

            #Add the classes according to the requirement
            for n in custom:
                try:
                    if n['label'] == 'bag':
                        num_ids.append(1)
                    elif n['label'] == 'belt':
                        num_ids.append(2)
                    elif n['label'] == 'boots':
                        num_ids.append(3)
                    elif n['label'] == 'footwear':
                        num_ids.append(4)
                    elif n['label'] == 'outer':
                        num_ids.append(5)
                    elif n['label'] == 'dress':
                        num_ids.append(6)
                    elif n['label'] == 'sunglasses':
                        num_ids.append(7)
                    elif n['label'] == 'pants':
                        num_ids.append(8)
                    elif n['label'] == 'top':
                        num_ids.append(9)
                    elif n['label'] == 'shorts':
                        num_ids.append(10)
                    elif n['label'] == 'skirt':
                        num_ids.append(11)
                    elif n['label'] == 'headwear':
                        num_ids.append(12)
                    elif n['label'] == 'scarf/tie':
                        num_ids.append(13)
                except:
                    pass

            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "custom",
                image_id=a['filename'],
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a custom dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "custom":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = image_info['num_ids']	
        #print("Here is the numID",num_ids)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        num_ids = np.array(num_ids, dtype=np.int32)	
        return mask, num_ids#.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32), 

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "custom":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(args.dataset, "val")
    dataset_val.prepare()

    # Pelatihan Model
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=150,
                layers='heads')


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect custom objects.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/custom/dataset/",
                        help='Directory of the custom dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    if args.command == "train":
        config = CustomConfig()
    else:
        class InferenceConfig(CustomConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 4
        config = InferenceConfig()
    config.display()

    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    if args.weights.lower() == "new":	
        print("weight path entered")	
        print(NEW_WEIGHTS_PATH)	
        weights_path = NEW_WEIGHTS_PATH
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
