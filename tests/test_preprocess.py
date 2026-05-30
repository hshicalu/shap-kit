import pandas as pd
from pandas.testing import assert_frame_equal

from preprocess import preprocess


def make_train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3],
            "Survived": [0, 1, 1],
            "Pclass": [3, 1, 3],
            "Name": [
                "Braund, Mr. Owen Harris",
                "Cumings, Mrs. John Bradley",
                "Heikkinen, Miss. Laina",
            ],
            "Sex": ["male", "female", "female"],
            "Age": [22.0, 38.0, 26.0],
            "SibSp": [1, 1, 0],
            "Parch": [0, 0, 0],
            "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282"],
            "Fare": [7.25, 71.2833, 7.925],
            "Cabin": [None, "C85", None],
            "Embarked": ["S", "C", "S"],
        }
    )


def test_preprocess_creates_numeric_features_without_mutating_input() -> None:
    train = make_train_frame()
    original = train.copy(deep=True)

    processed = preprocess(train, train)

    assert_frame_equal(train, original)
    assert "Name" not in processed
    assert "Ticket" not in processed
    assert "Cabin" not in processed
    assert processed["Sex"].tolist() == [0, 1, 1]
    assert processed["Salutation"].tolist() == [1, 3, 2]
    assert processed["Embarked"].tolist() == [0, 1, 0]
    assert processed["FamilySize"].tolist() == [2, 2, 1]
    assert processed["IsAlone"].tolist() == [0, 0, 1]


def test_preprocess_uses_training_statistics_for_missing_values() -> None:
    train = make_train_frame()
    test = pd.DataFrame(
        {
            "PassengerId": [4],
            "Pclass": [2],
            "Name": ["Allen, Dr. William Henry"],
            "Sex": ["male"],
            "Age": [None],
            "SibSp": [0],
            "Parch": [0],
            "Ticket": ["373450"],
            "Fare": [None],
            "Cabin": [None],
            "Embarked": [None],
        }
    )

    processed = preprocess(test, train)

    assert processed.loc[0, "Age"] == train["Age"].mean()
    assert processed.loc[0, "Fare"] == train["Fare"].mean()
    assert processed.loc[0, "Embarked"] == 0
    assert processed.loc[0, "Salutation"] == 5
    assert processed.loc[0, "Ticket_Lett"] == 3
    assert processed.loc[0, "Cabin_Lett"] == 0
