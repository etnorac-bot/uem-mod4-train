from ..features.feature_engineering import feature_engineering
from pathlib import Path
import pickle
import os

def make_datasets(path_train, path_test, model_config):
    """
        Función que permite crear los set usados para el entrenamiento y
        testeo del modelo.

        Args:
            path_train (str):  Ruta hacia el set de train.
            path_test (str): Ruta hacia el set de test.

        Kwargs:
           model_type (str): tipo de modelo usado.

        Returns:
           DataFrame, DataFrame. Datasets de train y test para el modelo.
    """

    print('---> Getting data')
    X_train, y_train = get_raw_data_from_local(path_train)
    X_test, y_test = get_raw_data_from_local(path_test)

    print('---> Feature engineering')
    X_train_hog, X_test_hog = feature_engineering(X_train, X_test, model_config)

    return X_train_hog, y_train, X_test_hog, y_test



def get_raw_data_from_local(path):

    """
        Función para obtener los datos originales desde local

        Args:
           path (str):  Ruta hacia los datos.

        Returns:
           (X,y ). X numpy.array de imagen y clase a la que pertenece la imagen.
    """

    with open(Path(path), 'rb') as f:
        (X, y) = pickle.load(f)

    return X, y
