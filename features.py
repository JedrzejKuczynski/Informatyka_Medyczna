import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import color, io, measure, morphology


def threshold_and_label(file, threshold):
    img = io.imread(file)  # Wczytanie zdjęcia
    img_gray = color.rgb2gray(img)  # Przekonwertowanie na monochromatyczny
    mask = (img_gray < threshold)  # Progowanie
    labels = measure.label(mask)  # Etykietowanie
    return labels


def subleaves(leaf_img, mask):
    eroded_img = morphology.binary_erosion(leaf_img, mask)  # Erozja
    eroded_img_labelled = measure.label(eroded_img)  # Etykietowanie
    eroded_img_regions = measure.regionprops(eroded_img_labelled)  # Regiony
    subleaves_number = len(eroded_img_regions)  # Liczba regionów
    return subleaves_number


def contour_histogram(contour_points, list_of_angles):
    hist_data = []  # Dane do histogramu --> wszystkie kąty razem
    for angle in list_of_angles:
        angles_in_degrees = []  # Lista przechowująca aktualne kąty
        contour_beginning = contour_points[:angle, :]  # Odpowiedni początek
        contour_end = contour_points[-angle:, :]  # Odpowiedni koniec
        # I robimy kółeczko
        contour_hist = np.insert(contour_points, 0, contour_end, axis=0)
        contour_hist = np.append(contour_hist, contour_beginning, axis=0)
        for i in range(angle, len(contour_hist) - angle):
            middle_point = contour_hist[i]  # Punkt środkowy
            left_point = contour_hist[i-angle]  # Punkt lewy
            right_point = contour_hist[i+angle]  # Punkt prawy

            # print(left_point, middle_point, right_point)

            # Wektory do obliczeń
            middle_right = right_point - middle_point
            middle_left = left_point - middle_point

            # print(middle_right, middle_left)

            # No i wzorki
            dot_product = np.dot(middle_right, middle_left)
            middle_right_norm = np.linalg.norm(middle_right)
            middle_left_norm = np.linalg.norm(middle_left)

            cosine = dot_product / (middle_right_norm * middle_left_norm)

            # Poprawka zaokrąglenia komputerowego i te sprawy
            if cosine < -1 or cosine > 1:
                cosine = np.round(cosine, 0)

            # Przerabiamy na kąt
            points_angle = np.arccos(cosine)
            points_angle = np.degrees(points_angle)
            points_angle = np.round(points_angle, 0)

            angles_in_degrees.append(points_angle)  # Dodajemy do aktualnej

        hist_data.append(angles_in_degrees)  # Dodajemy do wspólnej

    hist_data = np.concatenate(hist_data)  # Łączymy wspólną
    histogram = np.histogram(hist_data, bins=20)  # Liczymy histogram

    # plt.hist(hist_data, bins=20)
    # plt.show()
    # draw_contour(leaf_img_closed, contour_points)

    return histogram

# ------------- Kasiowa sekcja ----------------- #


def calculate_area(leaf):
    return leaf.area


def calculate_contour(leaf_img_closed):
    contours = measure.find_contours(leaf_img_closed, 0.8)
    longest = 0

    if len(contours) > 1:
        for contour in contours:
            if len(contour) > longest:
                longest = len(contour)
                longest_points = contour
    else:
        longest = len(contours[0])
        longest_points = contours[0]

    return longest, longest_points


def draw_contour(leaf_img, contour):
    io.imshow(leaf_img, interpolation='nearest', cmap=plt.cm.gray)
    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
    plt.show()


def calculate_area_to_contour(area, contour):
    return round(area/contour, 5)


def calculate_tail(leaf_img_closed):
    leaf_open = morphology.binary_opening(leaf_img_closed, morphology.disk(3))
    tail = np.bitwise_xor(leaf_img_closed, leaf_open)
    return np.sum(morphology.binary_opening(tail))


# Słownik przechowujący wszystkie liście i ich cechy
# Klucz to gatunek, wartość to lista krotek o postaci (id, pole, ... )
features = {}
threshold = 0.7  # Granica do progowania obrazu

for root, dirs, files in os.walk("./leafsnap-subset1", topdown=False):
    species = root.split(os.sep)[-1]  # Nazwa gatunku zawsze na końcu
    index = 0
    if species != "leafsnap-subset1":
        # Gatunek --> lista krotek z cechami kolejnych liści
        features[species] = []
    for name in files:
        filepath = os.path.join(root, name)
        img_labelled = threshold_and_label(filepath, threshold)
        region_props = measure.regionprops(img_labelled)  # Regiony
        width = len(img_labelled[0, :])  # Szerokość danego zdjęcia w pikselach
        height = len(img_labelled[:, 0])  # Wysokość danego zdjęcia w pikselach
        images_found = []  # Lista przechowująca potencjalne liście

        for region in region_props:
            bound_box = region.bbox  # Bounding box regionu
            if bound_box[2] < 0.88 * height and bound_box[3] < 0.88 * width:
                if bound_box[0] < 0.6 * height and bound_box[1] < 0.6 * width:
                    if region.area >= 4000:
                        # Jezeli znaleziono obiekt w miarę na środku
                        # i o odpowiedniej wielkości
                        images_found.append(region)  # Dodaj go na listę

        # 2 zdjęcia w Carya glabra i 2 w Cornus florida odmiawiały współpracy
        # dlatego istnieje poniższy if

        if len(images_found) == 1:  # Jeżeli znaleziono tylko 1 taki obiekt

            # Ekstrakcja chech z liscia
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
            disk10_mask = morphology.disk(10)  # Dyskowa maska o promieniu 10
            subleaves_dict = subleaves(leaf_img, disk10_mask)

            # 5. Ząbkowatość
            angles = [3, 5, 10, 15, 30, 50]
            histogram = contour_histogram(contour_points, angles)

            # Podsumowanie znalezionych cech
            # print(species, index, area, area_to_contour, tail)
            # features[species].append((index, area, area_to_contour, tail))

        index += 1  # Zwiekszanie numeru kolejnych lisci w obrebie gatunku
