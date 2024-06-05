if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import mlflow
import pickle

# from mlflow.entities import ViewType
# from mlflow.tracking import MlflowClient
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error

EXPERIMENT_NAME = "green-cab-linear-regression-model"

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(EXPERIMENT_NAME)



@transformer
def transform(data, *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your transformation logic here
    model, vectorizer = data

    vectorizer_path = "dict_vectorizer.pkl"
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    serializedVectorizer = pickle.dumps(vectorizer)
    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "linear model for taxis")
        mlflow.log_artifact(vectorizer_path,"dict vectorizer")

    return model