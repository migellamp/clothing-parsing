import tkinter.font as tkFont
import json
import skimage.io
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import os
from skimage import measure
from descartes import PolygonPatch
import alphashape
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
import cv2


class SimpleConfig(Config):
    NAME = "coco_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 81

class CustomConfig(Config):
    """Configuration for training on the custom dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "custom"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 13  # Background + number of classes (Here, 3)

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


def get_max_str_index(lst):
    return max(enumerate(lst), key=lambda x: len(x[1]))[0]

def search(arr, N, x):
    for i in range(0, N):
        if (arr[i] == x):
            return i
    return -1

def searchArray(value, array):
    newArray = []
    for i in range(len(array)):
        if(array[i].class_name == value):
            newArray.push(array[i])
    return newArray

def changeFormat(rp):
    getintoarray = []
    for i in range(rp.shape[0]):
        getintoarray.append((rp[i][1], rp[i][0]))
    return getintoarray

class centroidAllPixel:
    def __init__(self, className, centroids):
        self.className = className
        self.centroids = centroids

def segm_size(segm):
    try:
        height = segm.shape[0]
        width = segm.shape[1]
    except IndexError:
        raise

    return height, width

def merged_mask(masks):
    n= masks.shape[2]
    
    if n!=0:        
        merged_mask = np.zeros((masks.shape[0], masks.shape[1]))
        for i in range(n):
            merged_mask+=masks[...,i]
        merged_mask=np.asarray(merged_mask,dtype=np.uint8)   
        return merged_mask
    return masks[:,:,0]

def check_size(eval_segm, gt_segm):
    h_e, w_e = segm_size(eval_segm)
    h_g, w_g = segm_size(gt_segm)

    if (h_e != h_g) or (w_e != w_g):
        raise EvalSegErr("DiffDim: Different dimensions of matrices!")

class EvalSegErr(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def compute_acc(mask1, mask2):
    predict_mask = merged_mask(mask1)
    gt_mask = merged_mask(mask2)
    check_size(predict_mask, gt_mask)

    N_p = np.sum(gt_mask)
    N_n = np.sum(np.logical_not(gt_mask))

    TP = np.sum(np.logical_and(predict_mask, gt_mask))
    TN = np.sum(np.logical_and(np.logical_not(predict_mask), np.logical_not(gt_mask)))

    accuracy_ = ((TP + TN) / (N_p + N_n)) * 100

    return accuracy_

def compute_iou(predict_mask, gt_mask):
    if predict_mask.shape[2]==0:
        return 0
    mask1 = merged_mask(predict_mask)
    mask2 = merged_mask(gt_mask)

    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    #print("Iou 2 : ",iou_score)
    return iou_score * 100

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the custom dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes according to the numbe of classes required to detect
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

        # Train or validation dataset?
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
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            #labelling each class in the given image to a number

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

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "custom",
                image_id=a['filename'],  # use file name as a unique image id
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

    def load_mask_bool(self, image_id):
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
                        dtype=np.bool)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = False

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        num_ids = np.array(num_ids, dtype=np.int32)	
        return mask#.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32), 
    

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "custom":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def RunClothingSegmentation():
    global r
    global class_names
    class InferenceConfig(CustomConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()

    model = modellib.MaskRCNN(mode="inference", model_dir=os.getcwd(), config=config)
    model.load_weights("./resource/weight/mask_rcnn_custom_0080.h5", by_name=True)

    class_names = ['BG', 'bag', 'belt', 'boots', 'footwear', 'outer', 'dress', 'sunglasses', 'pants', 'top', 'shorts', 'skirt', 'headwear', 'scraf/tie']

    image = skimage.io.imread(filename,plugin='matplotlib')
    results = model.detect([image], verbose=1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], title="ClothingSegmentation.jpeg")

def RunBodySegmentation():
    global r1
    global class_names_orang
    config1 = SimpleConfig()

    model1 = modellib.MaskRCNN(mode="inference", config=config1, model_dir=os.getcwd())
    model1.load_weights("./resource/weight/mask_rcnn_coco.h5", by_name=True)

    class_names_orang = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                    'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard',
                    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                    'teddy bear', 'hair drier', 'toothbrush']
    image = skimage.io.imread(filename,plugin='matplotlib')
    result1 = model1.detect([image], verbose=1)
    r1 = result1[0]
    visualize.display_instances(image, r1['rois'], r1['masks'],r1['class_ids'], class_names_orang, r1['scores'], title="BodySegmentation.jpeg")

def RunSuperpixel():
    image = skimage.io.imread(filename,plugin='matplotlib')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageSkin = cv2.imread(filename)
    skin = cv2.cvtColor(imageSkin, cv2.COLOR_BGR2YCR_CB)

    plt.figure(figsize=(10, 10))
    _, ax = plt.subplots(1, figsize=(10,10))

    setsegmentnumber = 900
    setsigma = 2
    segments1 = slic(image, n_segments=setsegmentnumber, compactness=20, sigma=setsigma, convert2lab=True)
    flag = 1
    numberofTrue = 0
    allPixelDesc = []
    allClassName =[]
    pixelNumber = 0
    saveAllPredictedMask = []*2
    AllChangedPixel = []

    color = dict(
        top=(210/255,126/255,89/255),
        footwear=(235/255, 148/255,0),
        outer=(125/255,21/255,89/255),
        skirt=(89/255,144/255,210/255),
        dress=(102/255,204/255,0),
        pants=(0,255/255,255/255),
        belt=(255/255,0,255/255),
        headwear=(255/255,153/255,153/255), 
        boots=(255/255,0,0),
        bag=(255/255,1,124/255),
        shorts=(0,0,1),
        sunglasses=(0.3,1,0.8),
        scarf=(0.4,0.5,1),
        no_class=(0,0,0),
    )

    ClassIndex = []

    for i in r1['class_ids']:
        ClassIndex.append(class_names_orang[i])

    for region in regionprops(segments1):
        cx, cy = region.centroid
        mask_num = len(r['class_ids'])

        indexPerson = ClassIndex.index("person")
        maskBody = r1['masks'][:,:,indexPerson]
        for number in range(mask_num):
            maskClothing = r['masks'][:,:,number]
            if(maskClothing[int(cx)][int(cy)] == True):
                each_classnames = class_names[r['class_ids'][number]]
                if(pixelNumber > 0):
                    if(allPixelDesc[pixelNumber-1][0] == region):
                        continue
                c1 = str(gray[int(cx),int(cy)])
                status = "predicted"
                if(each_classnames == "scraf/tie"):
                    allPixelDesc.append((region, "scraf", c1, status, pixelNumber,cx,cy))
                else :
                    allPixelDesc.append((region, each_classnames, c1, status, pixelNumber,cx,cy))
                pixelNumber+=1
                numberofTrue+=1
        if(numberofTrue<1):
            if (maskBody[int(cx)][int(cy)] == True):
                status = "unpredicted"
                allPixelDesc.append((region, "no_class", str(gray[int(cx),int(cy)]),status, pixelNumber,cx,cy))
                pixelNumber+=1
        numberofTrue = 0
        flag+=1

    print("start processing ... ")

    for i in r['class_ids']:
        allClassName.append(class_names[i])

    print(allPixelDesc)

    for i in range(len(allPixelDesc)):
        raw_coordinate = allPixelDesc[i][0].coords
        new_coordinate = changeFormat(np.array(raw_coordinate))
        cx, cy = allPixelDesc[i][0].centroid    
        lower = np.array([0, 133, 77], dtype = "uint8")
        upper = np.array([235, 173, 127], dtype = "uint8")
        skinArea = cv2.inRange(skin[int(cx)][int(cy)], lower,upper) 
        indexPerson = ClassIndex.index("person")
        maskBody = r1['masks'][:,:,indexPerson] 

        if isPredicted == False:
            if(allPixelDesc[i][3] == "predicted"):
                if(len(saveAllPredictedMask) == 0):
                    saveAllPredictedMask.append((allPixelDesc[i][1], new_coordinate))
                    continue
                else:
                    if(allPixelDesc[i][1] not in np.array(saveAllPredictedMask,dtype=object)):
                        saveAllPredictedMask.append((allPixelDesc[i][1], new_coordinate))
                    else:
                        for j in range(len(saveAllPredictedMask)):
                            if(saveAllPredictedMask[j][0] == allPixelDesc[i][1]):
                                saveAllPredictedMask[j][1].extend(new_coordinate)
                    continue

        if(allPixelDesc[i][3] == "unpredicted" and skinArea[1]==0):
            for j in range(len(allPixelDesc)):
                if(allPixelDesc[i][2] == allPixelDesc[j][2] and allPixelDesc[j][3] == "predicted" and allPixelDesc[j][1] == allPixelDesc[i-1] and allPixelDesc[j][1] == allPixelDesc[i+1]):
                    allPixelDesc[i] = (allPixelDesc[i][0], allPixelDesc[j][1], allPixelDesc[j][2], "predicted", allPixelDesc[i][4])
                    if(allPixelDesc[i][4] not in AllChangedPixel):
                        AllChangedPixel.append(allPixelDesc[i][4])
                    if(allPixelDesc[j][1] not in np.array(saveAllPredictedMask,dtype=object)):
                        saveAllPredictedMask.append((allPixelDesc[j][1], new_coordinate))
                    else:
                        for k in range(len(saveAllPredictedMask)):
                            if(saveAllPredictedMask[k][0] == allPixelDesc[j][1]):
                                saveAllPredictedMask[k][1].extend(new_coordinate)   
                if(j < len(allPixelDesc)):
                    if(len(allPixelDesc)!= i+1):
                        if(allPixelDesc[i][2] == allPixelDesc[j][2] and allPixelDesc[j][3] == "predicted" and allPixelDesc[j][1] == allPixelDesc[i+1][1]):
                            allPixelDesc[i] = (allPixelDesc[i+1][0], allPixelDesc[j][1], allPixelDesc[j][2], "predicted", allPixelDesc[i][4])
                            if(allPixelDesc[i][4] not in AllChangedPixel):
                                AllChangedPixel.append(allPixelDesc[i][4])
                            if(allPixelDesc[i][1] not in np.array(saveAllPredictedMask,dtype=object)):
                                saveAllPredictedMask.append((allPixelDesc[i][1], new_coordinate))
                            else:
                                for k in range(len(saveAllPredictedMask)):
                                    if(saveAllPredictedMask[k][0] == allPixelDesc[i][1]):
                                        saveAllPredictedMask[k][1].extend(new_coordinate)
                if(allPixelDesc[i][2] == allPixelDesc[j][2] and allPixelDesc[j][3] == "predicted" and allPixelDesc[j][1] == allPixelDesc[i-1][1]):
                    allPixelDesc[i] = (allPixelDesc[i-1][0], allPixelDesc[j][1], allPixelDesc[j][2], "predicted", allPixelDesc[i][4])
                    if(allPixelDesc[i][4] not in AllChangedPixel):
                        AllChangedPixel.append(allPixelDesc[i][4])
                    if(allPixelDesc[i][1] not in np.array(saveAllPredictedMask,dtype=object)):
                        saveAllPredictedMask.append((allPixelDesc[i][1], new_coordinate))
                    else:
                        for k in range(len(saveAllPredictedMask)):
                            if(saveAllPredictedMask[k][0] == allPixelDesc[i][1]):
                                saveAllPredictedMask[k][1].extend(new_coordinate)

    for i in range(len(AllChangedPixel)):
        if(i<len(AllChangedPixel)):
            if(len(AllChangedPixel)!= i+1):
                print("range " + str(AllChangedPixel[i]) + "-" + str(AllChangedPixel[i+1]))
                for j in range(len(allPixelDesc)):
                    if(allPixelDesc[j][3] == "unpredicted"):
                        raw_coordinate = allPixelDesc[j][0].coords
                        new_coordinate = changeFormat(np.array(raw_coordinate))
                        cx = allPixelDesc[j][5]
                        cy = allPixelDesc[j][6]
                        lower = np.array([0, 133, 77], dtype = "uint8")
                        upper = np.array([235, 173, 127], dtype = "uint8")
                        skinArea = cv2.inRange(skin[int(cx)][int(cy)], lower,upper)
                        if(int(allPixelDesc[j][4])>int(AllChangedPixel[i]) and int(allPixelDesc[j][4]) < int(AllChangedPixel[i+1])
                        and allPixelDesc[AllChangedPixel[i]][1] == allPixelDesc[AllChangedPixel[i+1]][1] and skinArea[1]==0):
                            if(maskBody[int(cx)][int(cy)] == True):
                                if(allPixelDesc[AllChangedPixel[i]][1] not in np.array(saveAllPredictedMask,dtype=object)):
                                    saveAllPredictedMask.append((allPixelDesc[AllChangedPixel[i]][1], new_coordinate))
                                else:
                                    for k in range(len(saveAllPredictedMask)):
                                        if(saveAllPredictedMask[k][0] == allPixelDesc[AllChangedPixel[i]][1]):
                                            saveAllPredictedMask[k][1].extend(new_coordinate)   

    for i in range(len(saveAllPredictedMask)):
        print(saveAllPredictedMask[i][0])


    for i in range(len(saveAllPredictedMask)):
        alpha_shape = alphashape.alphashape(saveAllPredictedMask[i][1], 1)
        if isBBOXActive == True:
            if(alpha_shape.type == "Polygon"):
                xx,yy = alpha_shape.exterior.coords.xy
                xs = [min(xx),max(xx),max(xx),min(xx),min(xx)]
                ys = [min(yy),min(yy),max(yy),max(yy),min(yy)]
                ax.plot(xs, ys, color=color[saveAllPredictedMask[i][0]],linestyle='dashed')
            if(alpha_shape.type == "MultiPolygon"):
                for j in range(len(list(alpha_shape))):
                    xx, yy = alpha_shape[j].exterior.coords.xy
                    xs = [min(xx),max(xx),max(xx),min(xx),min(xx)]
                    ys = [min(yy),min(yy),max(yy),max(yy),min(yy)]
                    ax.plot(xs, ys, color=color[saveAllPredictedMask[i][0]], linestyle='dashed')
        ax.add_patch(PolygonPatch(alpha_shape, alpha=0.5,color=color[saveAllPredictedMask[i][0]],ec='black'))
    
    if isActive == True:
        plt.imshow(image.astype(np.uint8))
        plt.imshow(mark_boundaries((image).astype(np.uint8), segments1))
    else:
        plt.imshow(image.astype(np.uint8))
    plt.axis('off')
    plt.savefig("Superpixel.jpeg", bbox_inches='tight') 

def body():
    RunBodySegmentation()
    images = Image.open("BodySegmentation.jpeg")
    resize_image = images.resize((234,356))
    image = ImageTk.PhotoImage(resize_image)
    imageboxOutput.config(image=image)
    imageboxOutput.image = image
    imageboxOutput.place(y=77,x=542,width=234,height=356)

def clothing():
    RunClothingSegmentation()
    images = Image.open("ClothingSegmentation.jpeg")
    resize_image = images.resize((234,356))
    image = ImageTk.PhotoImage(resize_image)
    imageboxOutput.config(image=image)
    imageboxOutput.image = image
    imageboxOutput.place(y=77,x=542,width=234,height=356)

def superpixel():
    RunSuperpixel()
    images = Image.open("Superpixel.jpeg")
    resize_image = images.resize((234,356))
    image = ImageTk.PhotoImage(resize_image)
    imageboxOutput.config(image=image)
    imageboxOutput.image = image
    imageboxOutput.place(y=77,x=542,width=234,height=356)

    colormaps = tk.Label(root)
    imagess = Image.open('./resource/colormap.png')
    resize_image = imagess.resize((129,287))
    imagesss = ImageTk.PhotoImage(resize_image) 
    colormaps.config(image=imagesss)
    colormaps.image = imagesss
    colormaps.place(y=77,x=818)

def setMaskBoundaries():
    global isActive
    if isActive == True:
        isActive = False
    else: 
        isActive = True

def setPredicted():
    global isPredicted
    if isPredicted == True:
        isPredicted = False
    else: 
        isPredicted = True

def setBBx():
    global isBBOXActive
    if isBBOXActive == True:
        isBBOXActive = False
    else: 
        isBBOXActive = True

def upload_file():
    global img
    global filename
    f_types = [('Jpg Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    images = Image.open(filename)
    resize_image = images.resize((234,356))
    image = ImageTk.PhotoImage(resize_image)
    imageboxUpload.config(image=image)
    imageboxUpload.image = image
    imageboxUpload.place(y=77,x=220,width=234,height=356)
    
def exit():
    root.destroy()
    
isActive = False
isPredicted = False
isBBOXActive = False

root = tk.Tk()
root.title("Clothing Segmentation")
width=1000
height=600
screenwidth = root.winfo_screenwidth()
screenheight = root.winfo_screenheight()
alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
root.geometry(alignstr)
root.resizable(width=False, height=False)

firstInputFrame=tk.Label(root)
firstInputFrame["bg"] = "#f6f6f5"
ft = tkFont.Font(family='Times',size=10)
firstInputFrame["font"] = ft
firstInputFrame["fg"] = "#333333"
firstInputFrame["justify"] = "center"
firstInputFrame["text"] = ""
firstInputFrame.place(x=188,y=40,width=300,height=400)

firstOutputFrame=tk.Label(root)
firstOutputFrame["bg"] = "#f6f6f5"
ft = tkFont.Font(family='Times',size=10)
firstOutputFrame["font"] = ft
firstOutputFrame["fg"] = "#333333"
firstOutputFrame["justify"] = "center"
firstOutputFrame["text"] = ""
firstOutputFrame.place(x=512,y=40,width=300,height=400)

secondInputFrame=tk.Label(root)
secondInputFrame["bg"] = "#cccccc"
ft = tkFont.Font(family='Times',size=10)
secondInputFrame["font"] = ft
secondInputFrame["fg"] = "#333333"
secondInputFrame["justify"] = "center"
secondInputFrame["text"] = ""
secondInputFrame.place(x=193,y=77,width=288,height=356)

secondOutputFrame=tk.Label(root)
secondOutputFrame["bg"] = "#cccccc"
ft = tkFont.Font(family='Times',size=10)
secondOutputFrame["font"] = ft
secondOutputFrame["fg"] = "#333333"
secondOutputFrame["justify"] = "center"
secondOutputFrame["text"] = ""
secondOutputFrame.place(x=517,y=77,width=289,height=356)

inputLabel=tk.Label(root)
inputLabel["bg"] = "#f6f6f5"
ft = tkFont.Font(family='JetBrains',size=15)
inputLabel["font"] = ft
inputLabel["fg"] = "#333333"
inputLabel["justify"] = "center"
inputLabel["text"] = "INPUT IMAGE"
inputLabel.place(x=193,y=44,width=100,height=28)

outputLabel=tk.Label(root)
outputLabel["bg"] = "#f6f6f5"
ft = tkFont.Font(family='JetBrains',size=15)
outputLabel["font"] = ft
outputLabel["fg"] = "#333333"
outputLabel["justify"] = "center"
outputLabel["text"] = "OUTPUT"
outputLabel.place(x=517,y=44,width=60,height=28)

uploadButton=tk.Button(root)
uploadButton["bg"] = "#efefef"
ft = tkFont.Font(family='JetBrains',size=13)
uploadButton["font"] = ft
uploadButton["fg"] = "#000000"
uploadButton["justify"] = "center"
uploadButton["text"] = "Upload"
uploadButton.place(x=190,y=450,width=99,height=35)
uploadButton["command"] = lambda:upload_file()

ClothingButton=tk.Button(root)
ClothingButton["bg"] = "#efefef"
ft = tkFont.Font(family='JetBrains',size=13)
ClothingButton["font"] = ft
ClothingButton["fg"] = "#000000"
ClothingButton["justify"] = "center"
ClothingButton["text"] = "Clothing Segmentation"
ClothingButton.place(x=300,y=450,width=187,height=35)
ClothingButton["command"] = lambda:clothing()

bodyButton=tk.Button(root)
bodyButton["bg"] = "#efefef"
ft = tkFont.Font(family='JetBrains',size=13)
bodyButton["font"] = ft
bodyButton["fg"] = "#000000"
bodyButton["justify"] = "center"
bodyButton["text"] = "Body Segmentation"
bodyButton.place(x=190,y=500,width=296,height=35)
bodyButton["command"] = lambda:body()

superpixelButton=tk.Button(root)
superpixelButton["bg"] = "#efefef"
ft = tkFont.Font(family='JetBrains',size=13)
superpixelButton["font"] = ft
superpixelButton["fg"] = "#000000"
superpixelButton["justify"] = "center"
superpixelButton["text"] = "Superpixel Method"
superpixelButton.place(x=515,y=500,width=299,height=35)
superpixelButton["command"] = lambda:superpixel()

checkBox=tk.Checkbutton(root)
ft = tkFont.Font(family='JetBrains',size=13)
checkBox["font"] = ft
checkBox["fg"] = "#333333"
checkBox["justify"] = "center"
checkBox["text"] = "Apply Superpixel"
checkBox.place(x=515,y=449)
checkBox["offvalue"] = "0"
checkBox["onvalue"] = "1"
checkBox["command"] = lambda:setMaskBoundaries()

checkBox1=tk.Checkbutton(root)
ft = tkFont.Font(family='JetBrains',size=13)
checkBox1["font"] = ft
checkBox1["fg"] = "#333333"
checkBox1["justify"] = "center"
checkBox1["text"] = "Display Unpredicted Only"
checkBox1.place(x=645,y=449)
checkBox1["offvalue"] = "0"
checkBox1["onvalue"] = "1"
checkBox1["command"] = lambda:setPredicted()

checkBox2=tk.Checkbutton(root)
ft = tkFont.Font(family='JetBrains',size=13)
checkBox2["font"] = ft
checkBox2["fg"] = "#333333"
checkBox2["justify"] = "center"
checkBox2["text"] = "Display Bounding Box"
checkBox2.place(x=515,y=474)
checkBox2["offvalue"] = "0"
checkBox2["onvalue"] = "1"
checkBox2["command"] = lambda:setBBx()



# saveButton=tk.Button(root)
# saveButton["bg"] = "#efefef"
# ft = tkFont.Font(family='JetBrains',size=10)
# saveButton["font"] = ft
# saveButton["fg"] = "#000000"
# saveButton["justify"] = "center"
# saveButton["text"] = "Save Image"
# saveButton.place(x=510,y=500,width=299,height=35)
# saveButton["command"] = lambda:save()

imageboxOutput = tk.Label(root)
imageboxUpload = tk.Label(root)

root.mainloop()
