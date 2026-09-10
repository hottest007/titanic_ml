import joblib
import pandas as pd 




# test for presence of trained model

def test_model():

    assert joblib.load("model/titanic_logreg.pkl") is not None

# test for model prediction
def test_model_prediction():

    model = joblib.load("model/titanic_logreg.pkl")

    passenger = pd.DataFrame([{
        "Pclass": 1,
        "Sex": "female",
        "Age": 25,
        "SibSp": 3,
        "Parch": 1,
        "Fare": 25,
        "Embarked": "S",
        "Name": "Esther,miss, Josh",
        "Cabin": "C85"
    }])

    prediction = model.predict(passenger)
    probability = model.predict_proba(passenger)

    assert prediction[0] in [0, 1]
    assert probability.shape == (1, 2)
    assert 0 <= probability[0][1] <= 1