import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import color, io, measure, morphology


def threshold_and_label(file, threshold):
    img = io.imread(file)
    img_gray = color.rgb2gray(img)
    mask = (img_gray < threshold)
    labels = measure.label(mask)
    return labels


def subleaves(leaves, list_of_masks=[]):
    subleaves_dict = {}
    for key, value in leaves.items():
        subleaves_dict[key] = []
        for img in value:
            subleaves = []
            for mask in list_of_masks:
                eroded_img = morphology.binary_erosion(img, mask)
                eroded_img_labelled = measure.label(eroded_img)
                eroded_img_regions = measure.regionprops(eroded_img_labelled)
                subleaves_number = len(eroded_img_regions)
                subleaves.append(subleaves_number)
            subleaves_dict[key].append(subleaves)
    return subleaves_dict


leaves_dict = {}
threshold = 0.7

for root, dirs, files in os.walk("./leafsnap-subset1", topdown=False):
    species = root.split("/")[-1]
    if species != "leafsnap-subset1":
        leaves_dict[species] = []
    for name in files:
        filepath = os.path.join(root, name)
        img_labelled = threshold_and_label(filepath, threshold)
        region_props = measure.regionprops(img_labelled)
        width = len(img_labelled[0, :])
        height = len(img_labelled[:, 0])
        objects_found = []

        for region in region_props:
            bound_box = region.bbox
            if bound_box[2] < 0.88 * height and bound_box[3] < 0.88 * width:
                if bound_box[0] < 0.6 * height and bound_box[1] < 0.6 * width:
                    if region.area >= 4000:
                        objects_found.append(region.image)

        if len(objects_found) == 1:
            leaves_dict[species].append(objects_found[0])


disk_masks = [morphology.disk(5), morphology.disk(10), morphology.disk(15)]
subleaves_dict = subleaves(leaves_dict, disk_masks)
