import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_images(folder):
    """
    Function to load images as convert them to arrays (as gray scale images)
    :param folder: Parent folder location of the down_sampled_AR dataset
    :return: Array representation od images
    """
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    images = np.asarray(images)
    return images


def create_data_matrix(images):
    """
    Allocate space for all images in one data matrix.
        The size of the data matrix is
        ( w  * h, numImages )

        where,

        w = width of an image in the dataset.
        h = height of an image in the dataset.
        no. color channels as it is a grayscale image
    :param images: Images from input folder path - down_sampled_AR dataset face images
    :return: Data matrix of size 2600 * 198000, each row representing one image (flattened images)
    """

    print("Creating data matrix", end=" ... ")

    num_images = len(images)
    sz = images[0].shape
    data = np.zeros((num_images, sz[0] * sz[1]), dtype=np.float32)
    for i in list(range(0, num_images)):
        image = images[i].flatten()
        data[i, :] = image

    print("DONE")
    return data


def show_images(images, cols=1, titles=None):
    """
    Display a list of images in a single figure with matplotlib.

    :param images: List of np.arrays compatible with plt.imshow.
    :param cols Default = 1: Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).
    :param titles: List of titles corresponding to each image. Must have
            the same length as titles.
    :return: Figure with two columns, left representing test_class and right, determined_class
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)

    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]

    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    fig.suptitle('Test Class Sample Image vs Determined Class Sample Image', fontsize=60)
    plt.show()
