import numpy as np
from sklearn import preprocessing as pr


def category_mapper(config) -> np.ndarray:
    # In future, this will just return a dictionary with item-mappings. Now, just generate fake data

    # Define the item categories
    categories = np.array(['action', 'romance', 'comedy', 'thriller',
                           'drama', 'sci-fi', 'horror', 'adventure', 'animation', 'mystery'])

    # Set the random seed for reproducibility
    np.random.seed(42)

    item_map = {
        
    }

    return item_map, categories


def item_category_OHEvectors(num_items):
    return np.random.randint(low=0, high=2, size=(num_items, 10)
    )

print(item_category_OHEvectors(10).shape[1])

