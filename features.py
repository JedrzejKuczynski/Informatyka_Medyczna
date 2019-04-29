import matplotlib.pyplot as plt
import numpy as np
import os
import skimage


def img_labels(file, threshold):
    img = skimage.io.imread(file)
    img_gray = skimage.color.rgb2gray(img)
    mask = (img_gray < threshold)
    labels = skimage.measure.label(mask)
    return labels


# img_labelled_rgb = img_labels("leafsnap-subset1/crataegus_crus-galli/pi0062-07-2.jpg", 0.6)
# img_labelled_rgb = skimage.color.label2rgb(img_labelled_rgb)
# skimage.io.imshow(img_labelled_rgb)
# plt.show()


# Areal? BBox? Centroid?


counter_all = 0
counter_found = 0
threshold = 0.7

for root, dirs, files in os.walk("./leafsnap-subset1", topdown=False):
    for name in files:
        counter_all += 1
        filepath = os.path.join(root, name)
        # print(filepath)
        # print(counter)
        img_labelled = img_labels(filepath, threshold)
        region_props = skimage.measure.regionprops(img_labelled)
        # img_labelled_rgb = skimage.color.label2rgb(img_labelled)
        # skimage.io.imshow(img_labelled_rgb)
        # plt.show()
        counter_leaves = 0
        # print(img_labelled)
        width = len(img_labelled[0, :])
        height = len(img_labelled[:, 0])
        l = []

        for region in region_props:
            bounding_box = region.bbox
            # print("REGION: ", region.area, bounding_box, region.centroid)
            if bounding_box[2] < 0.88 * height and bounding_box[3] < 0.88 * width:
                if bounding_box[0] < 0.6 * height and bounding_box[1] < 0.6 * width:
                # if 300 < region.centroid[0] < 700 and 300 < region.centroid[1] < 700:
                    # print(region.centroid, "\t", region.area, "\t", region.bbox)
                    if region.area >= 4000:
                        # print("PRZESZLO: ", width, height, region.area, bounding_box, region.centroid)
                        # img_labelled_rgb = skimage.color.label2rgb(img_labelled)
                        # skimage.io.imshow(region.image)
                        # plt.show()
                        skimage.io.imshow(region.image)
                        plt.show()
                        counter_found += 1
                        counter_leaves += 1
                        l.append([width, height, region.area, bounding_box, region.centroid])

        if counter_leaves != 1:
            pass
            # print(filepath, counter_leaves)
            # img_labelled_rgb = skimage.color.label2rgb(img_labelled)
            # skimage.io.imshow(img_labelled_rgb)
            # plt.show()

print("ALL: ", counter_all, " FOUND: ", counter_found)

# skimage.io.imshow(img_labels)
# plt.show()
