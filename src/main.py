import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os
import math
from src import functions as fct
from sklearn.decomposition import PCA


TRAIN_DATA = "/home/harish/PycharmProjects/Eigenfaces - Face Recognition/down_sampled_AR"
training_images = os.listdir(TRAIN_DATA)
assert(len(training_images) > 0)


if __name__ == '__main__':

    # Loading the training data face images
    faces = fct.load_images(TRAIN_DATA)

    # Create matrix of data set
    faces_data_matrix = fct.create_data_matrix(faces)
    faces_mean = np.mean(faces_data_matrix, axis=0)

    # Rescaling data - Mean norm normalization
    data_rescaled = faces_data_matrix - faces_mean

    pca = PCA(0.995)
    matlab_score_equi = pca.fit_transform(data_rescaled)
    print('No. of Principal Components (95% variance)', pca.n_components_)

    reduced_dimensions_dot = np.dot(matlab_score_equi.T, data_rescaled)

    # Weighted EigenFaces
    weights_reduced_images = np.dot(data_rescaled, reduced_dimensions_dot.T)
    print('Shape of reduced images:', weights_reduced_images.shape)

    # ERROR Formulation
    error_count = 0
    weighted_diff_matrix = np.zeros(2599)  # Matrix to calculate euclidean distances
    display_matching_faces = []

    # Algo accuracy
    for i in range(1, 11):
        # Randomize test images (1 image for every 10 classes i.e., 260 images)
        start_counter = ((i - 1) * 260) + 1
        stop_counter = i * 260
        test_image_number = random.randint(start_counter, stop_counter)
        test_class = test_image_number // 26
        image_number = test_image_number % 26

        print("===========================================================")
        print('Iteration no.=', i, 'Sample test image number =', test_image_number)

        # Get weighted test image
        test_image_mean_norm = faces_data_matrix[test_image_number-1, :] - faces_mean
        weighted_test_image = np.dot(test_image_mean_norm, reduced_dimensions_dot.T)  # changed

        # Skip test image for training algorithm
        weight_diff_counter = 0
        for j in range(1, 2601):
            if j == test_image_number:
                continue
            weighted_difference = weights_reduced_images[(j-1), :] - weighted_test_image
            weighted_diff_matrix[weight_diff_counter] = math.sqrt(np.dot(weighted_difference, weighted_difference.T))
            weight_diff_counter += 1

        # Index position & class number of mismatched class
        class_index = np.argmin(weighted_diff_matrix)
        print("Minimum value's index:", class_index)
        determined_class = math.floor(class_index/26)

        # Counting errors for each mismatching class of faces
        print("Test class:", test_class, ", Determined class:", determined_class)
        print("===========================================================")

        if test_class != determined_class:
            error_count = error_count + 1
            continue

        display_matching_faces.append(faces_data_matrix[test_class, :].reshape(-1, 120))
        display_matching_faces.append(faces_data_matrix[determined_class, :].reshape(-1, 120))

    # Algorithm accuracy overall
    plt.show()
    print("Algorithm accuracy:", ((10 - error_count) * 10), "%")
    print(len(display_matching_faces))

    # Show test_class image and determined_class image
    fct.show_images(display_matching_faces, len(display_matching_faces)/2)
    plt.show()
    plt.pause(0.0001)
    cv2.destroyAllWindows()
