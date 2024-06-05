import pandas as pd
import scipy
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def transform(training_set:pd.DataFrame, *args, **kwargs):
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
    dv = DictVectorizer()
    features = ['PULocationID', 'DOLocationID']
    X_training_set = training_set[features]
    train_dicts = X_training_set.to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)
    target = 'duration'
    y_train = training_set[target].values
    model = LinearRegression()
    model.fit(X_train,y_train)
    print(model.intercept_)

    # X_val = None
    # if validation_set is not None:
    #     val_dicts = validation_set[training_set.columns].to_dict(orient='records')
    #     X_val = dv.transform(val_dicts)

    return (model,dv)