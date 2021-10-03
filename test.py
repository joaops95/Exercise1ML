import numpy as np
a = np.array([[1,2, 4, 4], [5, 5, 3,4]])
a = a.flatten()

a = a.reshape(2, 4)

data = np.asarray([0, 1, 2, 0])

random_action = np.unravel_index(np.argmax(data, axis=None), data.shape)[0]

print(random_action)


# print(a)