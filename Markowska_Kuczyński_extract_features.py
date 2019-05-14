import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import color, io, measure, morphology


parser = argparse.ArgumentParser(description="Skrypt obliczający \
cechy (pole powierzchni, obwód, liczbę podlistków itp.) z obrazu liścia, \
które będą wykorzystywane w uczeniu modelu.")
parser.add_argument("path", help="Ścieżka do katalogu z bazą obrazków")
parser.add_argument("outfile", help="Plik zawierający wyliczone cechy oraz \
nazwę klasy dla wszystkich obrazków z bazy")
args = parser.parse_args()


def threshold_and_label(file, threshold):
    img = io.imread(file)
    img_gray = color.rgb2gray(img)
    mask = (img_gray < threshold)
    labels = measure.label(mask)
    return labels


def subleaves(leaf_img, mask):
    eroded_img = morphology.binary_erosion(leaf_img, mask)
    eroded_img_labelled = measure.label(eroded_img)
    eroded_img_regions = measure.regionprops(eroded_img_labelled)
    subleaves_number = len(eroded_img_regions)
    return subleaves_number


def contour_histogram(contour_points, list_of_angles):
    hist_data = []
    for angle in list_of_angles:
        angles_in_degrees = []
        contour_beginning = contour_points[:angle, :]
        contour_end = contour_points[-angle:, :]

        # I robimy kółeczko
        contour_hist = np.insert(contour_points, 0, contour_end, axis=0)
        contour_hist = np.append(contour_hist, contour_beginning, axis=0)

        for i in range(angle, len(contour_hist) - angle):
            middle_point = contour_hist[i]
            left_point = contour_hist[i-angle]
            right_point = contour_hist[i+angle]

            # Wektory do obliczeń
            middle_right = right_point - middle_point
            middle_left = left_point - middle_point

            # No i wzorki
            dot_product = np.dot(middle_right, middle_left)
            middle_right_norm = np.linalg.norm(middle_right)
            middle_left_norm = np.linalg.norm(middle_left)

            cosine = dot_product / (middle_right_norm * middle_left_norm)

            # Poprawka zaokrąglenia komputerowego i te sprawy
            if cosine < -1 or cosine > 1:
                cosine = np.round(cosine, 0)

            points_angle = np.arccos(cosine)
            points_angle = np.degrees(points_angle)
            points_angle = np.round(points_angle, 0)

            angles_in_degrees.append(points_angle)

        hist_data.append(angles_in_degrees)

    hist_data = np.concatenate(hist_data)
    histogram = np.histogram(hist_data, bins=20)

    # plt.hist(hist_data, bins=20)
    # plt.show()
    # draw_contour(leaf_img_closed, contour_points)

    return histogram


def calculate_area(leaf):
    return leaf.area


def calculate_contour(leaf_img_closed):
    contours = measure.find_contours(leaf_img_closed, 0.8)
    longest = 0

    if len(contours) > 1: # Jesli wykryto wiecej niz 1 kontur to szukamy tego najdluzszego
        for contour in contours:
            if len(contour) > longest:
                longest = len(contour)
                longest_points = contour
    else: 
        longest = len(contours[0])
        longest_points = contours[0]

    return longest, longest_points


def calculate_area_to_contour(area, contour):
    return round(area/contour, 5)


def calculate_tail(leaf_img_closed):
    leaf_opened = morphology.binary_opening(leaf_img_closed, morphology.disk(4))
    tail = np.bitwise_xor(leaf_img_closed, leaf_opened) # Odejmowanie obrazu z ogonkiem i bez ogonka
    return np.sum(morphology.binary_opening(tail))


def draw_contour(leaf_img, contour):
    io.imshow(leaf_img, interpolation='nearest', cmap=plt.cm.gray)
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.show()



# Słownik przechowujący wszystkie liście i ich cechy
# Klucz to gatunek, wartość to lista krotek o postaci (pole, ... )
features_all = {}
threshold = 0.7  # Granica do progowania obrazu

for root, dirs, files in os.walk(args.path, topdown=False):
    species = root.split(os.sep)[-1]  # Nazwa gatunku zawsze na końcu
    if species and species != "leafsnap-subset1":
        features_all[species] = []

    
    for name in files:
        filepath = os.path.join(root, name)
        img_labelled = threshold_and_label(filepath, threshold)
        region_props = measure.regionprops(img_labelled)
        width = len(img_labelled[0, :])
        height = len(img_labelled[:, 0])
        images_found = []

        for region in region_props:
            bound_box = region.bbox
            if bound_box[2] < 0.88 * height and bound_box[3] < 0.88 * width:
                if bound_box[0] < 0.6 * height and bound_box[1] < 0.6 * width:
                    if region.area >= 4000:
                        images_found.append(region)

        # 2 zdjęcia w Carya glabra i 2 w Cornus florida odmiawiały współpracy
        # dlatego istnieje poniższy if

        if len(images_found) == 1:

            leaf = images_found[0]
            leaf_img = np.pad(leaf.image, 1, 'constant', constant_values=0)

            leaf_img_closed = morphology.binary_closing(leaf.image,
                                                        morphology.disk(5))

            leaf_img_closed = np.pad(leaf_img_closed, 1, 'constant',
                                     constant_values=0)

            # 1. Pole powierzchni
            area = calculate_area(leaf)

            # 2. Pole powierzchni do długości konturu
            contour_len, contour_points = calculate_contour(leaf_img_closed)
            area_to_contour = calculate_area_to_contour(area, contour_len)

            # 3. Długość ogonka
            tail = calculate_tail(leaf_img_closed)

            # 4. Liczba podlistków
            disk10_mask = morphology.disk(10)
            subleaves_number = subleaves(leaf_img, disk10_mask)

            # 5. Ząbkowatość
            angles = [3, 5, 10, 15, 30, 50]
            histogram = contour_histogram(contour_points, angles)

            features_species = [area, area_to_contour, tail,
                                subleaves_number, histogram]
            print(features_species)
            features_all[species].append(features_species)

np.savez(args.outfile, **features_all)
