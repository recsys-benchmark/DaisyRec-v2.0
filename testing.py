import numpy as np
import time

nrows = 20000

arr = np.ndarray(shape=(nrows, 4))

items = np.array(range(3000))

s = time.time()
for i in range(nrows):
    indices = np.random.randint(3000, size=4)
    arr[i] = items[indices]
e = time.time()


print(e-s)
