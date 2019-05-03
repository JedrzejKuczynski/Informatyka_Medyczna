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


def subleaves(leaves, list_of_masks=[]):
    subleaves_dict = {}  # Zwracany słownik: gatunek --> 3-elementowa lista
    for key, value in leaves.items():
        subleaves_dict[key] = []  # Dla każdego gatunku
        for img in value:  # Dla każdego zdjęcia
            subleaves = []  # Lista przechowująca liczbę znalezionych podliści
            for mask in list_of_masks:  # Dla każdej maski
                # Erozja --> Etykiety --> Regiony --> Liczba regionów
                eroded_img = morphology.binary_erosion(img, mask)
                eroded_img_labelled = measure.label(eroded_img)
                eroded_img_regions = measure.regionprops(eroded_img_labelled)
                subleaves_number = len(eroded_img_regions)
                subleaves.append(subleaves_number)  # I dodajemy
            subleaves_dict[key].append(subleaves)  # I na końcu do słownika
    return subleaves_dict


# TODO PONIŻSZA FUNKCJA. CO Z TYMI KONTURAMI JA SIĘ PYTAM??? GÓWNO <3

def contour_and_histogram(leaves, list_of_angles=[]):
    for key, value in leaves.items():
        for img in value:
            contours = measure.find_contours(img, 0.7)
            io.imshow(img)
            for i, contour in enumerate(contours):
                if len(contour) > 50:
                    pass
            plt.show()
    return

# ------------- Kasiowa sekcja -----------------


def calculate_area(leaf):
    return leaf.area


def calculate_contour(leaf_img_closed):
    contours = measure.find_contours(leaf_img_closed, 0.8)
    longest = 0

    if len(contours) > 1:
        for contour in contours:
            if len(contour) > longest:
                longest_len = len(contour)
                longest_points = contour
    else:
        longest_len = len(contours[0])
        longest_points = contours[0]

    return longest_len, longest_points


def draw_contour(image, contours):
    fig, ax = plt.subplots()
    ax.imshow(leaf_img, interpolation='nearest', cmap=plt.cm.gray)

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def calculate_area_to_contour(area, contour):
    return round(area/contour, 5)


leaves_dict = {}  # Słownik przechowujący wycięte liście
features = {}  # Słownik przechowujący wszystkie liście i ich cechy
# Klucz to gatunek, wartość to lista krotek o postaci (id, pole, ... )
threshold = 0.7  # Granica do progowania obrazu

for root, dirs, files in os.walk("./leafsnap-subset1", topdown=False):
    species = root.split(os.sep)[-1]  # Nazwa gatunku zawsze na końcu
    index = 0
    if species != "leafsnap-subset1":
        features[species] = []  # Gatunek --> lista krotek z cechami kolejnych liści
        leaves_dict[species] = []  # Gatunek --> lista wyciętych liści
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
            # DO KASI: TU BYM ZAPISAŁ SPADOWANY LIŚĆ. PYTANIE CZY PO CLOSING?
            # DO JJ: ale nam on zbędny. mi on zbędny, wiec sobie zapisz co tylko chcesz
            leaves_dict[species].append(images_found[0].image)

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

            # Podsumowanie znalezionych cech
            print(species, index, area, area_to_contour)

        index += 1  # Zwiekszanie numeru kolejnych lisci w obrebie gatunku


# Lista z maskami do erozji wykorzystywanej w znajdowaniu składowych liści
# Empirycznie sprawdzono, że trzy maski w zupełności wystarczą

disk_masks = [morphology.disk(5), morphology.disk(10), morphology.disk(15)]
# subleaves_dict = subleaves(leaves_dict, disk_masks)
# contour_and_histogram(leaves_dict)
