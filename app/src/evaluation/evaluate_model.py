from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime

def evaluate_model(model, X_test, y_test, timestamp, model_config):
    """
        Esta función permite realizar una evaluación del modelo entrenado
        y crear un diccionario con toda la información relevante del mismo

        Args:
           model (sklearn-object):  Objecto del modelo entrenado.
           X_test (DataFrame): Variables independientes en test.
           y_test (Series):  Variable dependiente en test.
           timestamp (float):  Representación temporal en segundos.
           model_name (str):  Nombre del modelo

        Returns:
           dict. Diccionario con la info del modelo
    """

    # creación del diccionario de info del modelo
    model_info = {}

    # info general del modelo
    model_info['_id'] = 'model_' + str(int(timestamp))
    model_info['name'] = 'model_' + str(int(timestamp))
    # fecha de entrenamiento (dd/mm/YY-H:M:S)
    model_info['date'] = datetime.now().strftime("%d/%m/%Y-%H:%M:%S")
    model_info['model_used'] = model_config['model_name']

    # Guardamos la configuracion del modelo para que se sepa como se entreno esencial para realizar inferencia
    # la inferencia extrae los parametros HOG que tiene que usar de este archivo
    model_info['model_config'] = model_config

    # métricas usadas
    # obtener predicciones usando el modelo entrenado
    y_pred = model.predict(X_test)

    model_info['model_metrics'] = {}
    model_info['model_metrics']['roc_auc_score'] = roc_auc_score(y_test, model.predict_proba(X_test),
                                                          average="weighted", multi_class="ovr")
    model_info['model_metrics']['accuracy_score'] = accuracy_score(y_test, y_pred)
    model_info['model_metrics']['precision_score'] = precision_score(y_test, y_pred, average="weighted")
    model_info['model_metrics']['recall_score'] = recall_score(y_test, y_pred, average="weighted")
    model_info['model_metrics']['f1_score'] = f1_score(y_test, y_pred, average="weighted")
    model_info['model_metrics']['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()

    # status del modelo (en producción o no)
    model_info['status'] = "none"

    return model_info



