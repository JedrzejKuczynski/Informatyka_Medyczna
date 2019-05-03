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


# TODO PONIŻSZA FUNKCJA. CO Z TYMI KONTURAMI JA SIĘ PYTAM???

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


leaves_dict = {}  # Słownik przechowujący wycięte liście
threshold = 0.7  # Granica do progowania obrazu

for root, dirs, files in os.walk("./leafsnap-subset1", topdown=False):
    species = root.split("/")[-1]  # Nazwa gatunku znajduje się zawsze na końcu
    if species != "leafsnap-subset1":
        leaves_dict[species] = []  # Gatunek --> lista wyciętych liści
    for name in files:
        filepath = os.path.join(root, name)
        img_labelled = threshold_and_label(filepath, threshold)
        region_props = measure.regionprops(img_labelled)  # Regiony
        width = len(img_labelled[0, :])  # Szerokość danego zdjęcia w pikselach
        height = len(img_labelled[:, 0])  # Wysokość danego zdjęcia w pikselach
        objects_found = []  # Lista przechowująca potencjalne liście

        for region in region_props:
            bound_box = region.bbox  # Bounding box regionu
            if bound_box[2] < 0.88 * height and bound_box[3] < 0.88 * width:
                if bound_box[0] < 0.6 * height and bound_box[1] < 0.6 * width:
                    if region.area >= 4000:
                        # Jezeli znaleziono obiekt w miarę na środku
                        # i o odpowiedniej wielkości

                        objects_found.append(region.image)  # Dodaj go na listę

        # 2 zdjęcia w Carya glabra i 2 w Cornus florida odmiawiały współpracy
        # dlatego istnieją poniższe dwie linijki

        if len(objects_found) == 1:  # Jeżeli znaleziono tylko 1 taki obiekt
            leaves_dict[species].append(objects_found[0])


# Lista z maskami do erozji wykorzystywanej w znajdowaniu składowych liści
# Empirycznie sprawdzono, że trzy maski w zupełności wystarczą

disk_masks = [morphology.disk(5), morphology.disk(10), morphology.disk(15)]
subleaves_dict = subleaves(leaves_dict, disk_masks)
contour_and_histogram(leaves_dict)
