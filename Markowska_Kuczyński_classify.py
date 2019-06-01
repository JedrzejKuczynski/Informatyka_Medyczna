import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from skimage import color, io, measure, morphology
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser(description="Skrypt obliczający \
cechy (pole powierzchni, obwód, liczbę podlistków itp.) z obrazu liścia, \
a następnie przewidujący gatunek, do którego należy z wykorzystaniem \
poprzednio nauczonego modelu.")
parser.add_argument("folderpath", help="Ścieżka do katalogu z obrazkami \
w formacie jpg")
args = parser.parse_args()


def load_and_prepare_data(filename):
    data = np.load(filename, allow_pickle=True)
    features_all = []
    targets = []

    for label in data.files:
        for features in data[label]:
            features_all.append(features)
            targets.append(label)

    features_all = np.array(features_all)
    targets = np.array(targets)

    features_all = StandardScaler().fit_transform(features_all)

    return features_all, targets


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
    hist, bin_edges = np.histogram(hist_data, bins=20, density=True)

    # plt.hist(hist_data, bins=20)
    # plt.show()
    # draw_contour(leaf_img_closed, contour_points)

    return hist


def calculate_area(leaf):
    return leaf.area


def calculate_contour(leaf_img_closed):
    contours = measure.find_contours(leaf_img_closed, 0.8)
    longest = 0

    # Jesli wykryto wiecej niz 1 kontur to szukamy tego najdluzszego
    if len(contours) > 1:
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
    tail = np.bitwise_xor(leaf_img_closed, leaf_opened)  # Odejmowanie obrazu z ogonkiem i bez ogonka
    return np.sum(morphology.binary_opening(tail))


def calculate_hu(leaf):
    return leaf.moments_hu


def draw_contour(leaf_img, contour):
    io.imshow(leaf_img, interpolation='nearest', cmap=plt.cm.gray)
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.show()


def extract_features(folderpath):
    features_all = []
    filenames = []

    threshold = 0.7  # Granica do progowania obrazu

    for filename in os.listdir(folderpath):
        filepath = os.path.join(folderpath, filename)
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
            hist = contour_histogram(contour_points, angles)

            # 6. Momenty Hu
            hu = calculate_hu(leaf)

            features_species = np.array([area, area_to_contour, tail,
                                        subleaves_number])
            features_species = np.concatenate((features_species, hist, hu))
            features_all.append(features_species)
            filenames.append(filename)

    features_all = StandardScaler().fit_transform(features_all)
    return features_all, filenames

clf = joblib.load("Markowska_Kuczyński_classifier.pkl")
features, filenames = extract_features(args.folderpath)


predictions = clf.predict(features)

for filename, prediction in zip(filenames, predictions):
    print(filename, prediction)
    
