from PIL import Image
import itertools
import numpy
import os

#create list of such lists: [name, Image]
def create_image_list(data_path):
    img_list = []
    for f in os.listdir(data_path):
        tmp = []
        tmp.append(f)
        tmp.append(Image.open(data_path + f))
        img_list.append(tmp)
    return img_list


# create list of such lists:[name, matrix, dtype]
def create_matrix_list(data_path):
    img_list = []
    for f in os.listdir(data_path):
        tmp = []
        tmp.append(data_path + f);
        tmp.append(numpy.array(Image.open(data_path + f)))
        img_list.append(tmp)
    return img_list


# find the same values of vector list and return it
def duplicate_elements(list):
    i = 0
    result = []
    result2 = []
    for i in range(0, len(list)):
        tmp = []
        tmp2 = []
        if not is_in_this(list[i], result):
            tmp.append(list[i])
            tmp2.append(list[i][0])
            for j in range(i + 1, len(list)):
                if numpy.array_equal(tmp[0][1],list[j][1]) and not is_in_this(list[j], tmp):
                    #print("between " + tmp[0][0] + " and "+ list[j][0])
                    tmp2.append(list[j][0])
                    tmp.append(list[j])
            result.append(tmp)
            result2.append(tmp2)
    return result2

# is obj in any list element
def is_in_this(obj, list):
    for i in list:
        if obj in i:
            return True
    return False

def print_duplicates(list):
    for elem in list:
        if len(elem) >= 2:
            print("Duplicates: ")
            for e in elem:
                print(e)

# ------------------------- DUPLICATE DETECTION PART END --------------------------

def img_gray(img):
    return numpy.average(img, weights=[0.299,0.587,0.114], axis=2)

def img_gray_from_obj(image):
    image = Image.fromarray(numpy.average(numpy.array(image), weights=[0.299,0.587,0.114], axis=2))
    return image

def resize_img(image, height=30, width=30):
    img = image.resize((height,width))
    img = numpy.array(img)
    row_res = numpy.array(img).flatten()
    col_res = numpy.array(img).flatten('F')
    return row_res, col_res

def intensity_diff(row_res, col_res):
    difference_row = numpy.diff(row_res)
    difference_col = numpy.diff(col_res)
    difference_row = difference_row > 0
    difference_col = difference_col > 0
    return numpy.vstack((difference_row, difference_col)).flatten()

def difference_score(image, height=30, width=30):
    gray = img_gray_from_obj(image)
    row_res, col_res = resize_img(gray, height, width)
    difference = intensity_diff(row_res, col_res)
    return difference

def hamming_distance(a, b):
    return (numpy.count_nonzero(a!=b)/len(a))

def difference_score_dict(image_list):
    dict = {}
    #duplicates = []
    for image in image_list:
        ds = difference_score(image[1])
        if image[0] not in dict:
            dict[image[0]] = ds
        #else:
            #duplicates.append((image, dict[image]))
    return dict

def find_similar(dict):
    similar = []
    for k1,k2 in itertools.combinations(dict,2):
        if hamming_distance(dict[k1],dict[k2]) < .10:
            similar.append((k1,k2))
    return similar

#------------Building a CNN to detect similar images

