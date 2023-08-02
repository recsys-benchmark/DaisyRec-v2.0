from daisy.utils.metrics import Intra_List_Distance, DiversityScore, FScore, Entropy
from daisy.utils.categorymap import item_category_OHEvectors
import numpy as np

topk = 10

pred_ur = np.array([
    [19,23,34,62,26,45,284,845,454,35],
    [1,2,3,4,5,6,7,8,9,10],
    [89,11,32,34,71,823,123,65,23,265],
])

item_cat_map = item_category_OHEvectors(1000, 10)

print(
    Entropy(pred_ur, item_cat_map)
)

