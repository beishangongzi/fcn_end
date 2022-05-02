# create by andy at 2022/5/2
# reference:
import numpy as np
import spectral
from spectral import save_rgb


def one2multi(y):
    # array_y = np.load(y)
    # print(array_y.shape)
    # print(np.max(array_y))
    # print(np.min(array_y))
    # print(np.unique(array_y))
    # new_y = np.zeros([5, 256, 256])
    # new_y[0, :, :] = (array_y == 0).reshape(256, 256)
    # new_y[1, :, :] = (array_y == 1).reshape(256, 256)
    # new_y[2, :, :] = (array_y == 2).reshape(256, 256)
    # new_y[3, :, :] = (array_y == 3).reshape(256, 256)
    # new_y[4, :, :] = (array_y == 4).reshape(256, 256)
    # print(new_y)

    new_y = np.zeros([256, 256, 5])
    new_y[:, :, 0] = (y == 0).reshape(256, 256)
    new_y[:, :, 1] = (y == 1).reshape(256, 256)
    new_y[:, :, 2] = (y == 2).reshape(256, 256)
    new_y[:, :, 3] = (y == 3).reshape(256, 256)
    new_y[:, :, 4] = (y == 4).reshape(256, 256)
    return new_y
    # print(new_y.shape)
    # save_rgb("test.jpg", new_y, colors=spectral.spy_colors)
    # save_rgb("test3.jpg", array_y, colors=spectral.spy_colors)


if __name__ == '__main__':
    one2multi("data/obt/imageMasks/0002_2.npy")
