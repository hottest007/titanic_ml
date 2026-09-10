from app.app import app
from fastapi.testclient import TestClient


client = TestClient(app)

def test_homepage():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Titanic API running"}


def test_predict():
    payload = {"Pclass": 1,
               "Sex": "female",
               "Age": 25,
               "SibSp": 3,
               "Parch": 1,
               "Fare": 25,
               "Embarked": "S",
               "Name": "Esther,miss, Josh",
               "Cabin": "C85"
               }


    response = client.post("/predict", json=payload)
    print("STATUS:", response.status_code)
    print("RESPONSE:", response.json())

    data = response.json()
    assert response.status_code == 200
    assert "prediction" in data
    assert "probability" in data
    assert data["prediction"] in [0,1]
    assert data["Outccome"] in ["Survived","Did not Survive"]
    