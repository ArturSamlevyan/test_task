from PIL import Image
import numpy as np
import os
from functions import *

data_path = 'data/'

#---- Find duplicates
img_list = create_matrix_list(data_path)
a = duplicate_elements(img_list)
print_duplicates(a)
#------------------------------------
print("\n\n")
#---- Find modification images
img_list_2 = create_image_list(data_path)
ds_dict = difference_score_dict(img_list_2)
sim = find_similar(ds_dict)
print("Modification images in couples(with duplicates): ")
print(sim)









