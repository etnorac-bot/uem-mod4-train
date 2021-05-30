from skimage.feature import hog
import numpy as np


def feature_engineering(X_train, X_test, model_config):
    """
        Función para encapsular la tarea de ingeniería de variables

        Args:
           X_train (np.array):  Dataset de train.
           X_test (np.array):  Dataset de test.

        Returns:
           Np.array, Np.array. Arrays de train y test para el modelo.
    """

    orient = model_config['HOG_orientation']
    ppc_list = model_config['HOG_ppc']

    X_train_n = (0.5 - X_train / 255)
    X_test_n = (0.5 - X_test / 255)

    print('------> Performing HOG extraction')
    print('---------> Performing HOG extraction TRAIN')
    X_train_hog = np.concatenate([np.concatenate([hog(xi, orientations=orient,
                                                      pixels_per_cell=(ppc, ppc),
                                                      cells_per_block=(1, 1),
                                                      visualize=False,
                                                      block_norm='L1')[np.newaxis, :] for xi in X_train_n[:, :, :]],
                                                      axis=0) for ppc in ppc_list], axis=1)
    print('---------> Performing HOG extraction TEST')
    X_test_hog = np.concatenate([np.concatenate([hog(xi, orientations=orient,
                                                     pixels_per_cell=(ppc, ppc),
                                                     cells_per_block=(1, 1),
                                                     visualize=False,
                                                     block_norm='L1')[np.newaxis, :] for xi in X_test_n[:, :, :]],
                                                     axis=0) for ppc in ppc_list], axis=1)

    print('------> HOG extraction done!')

    return X_train_hog, X_test_hog

