import numpy as np
import pandas as pd

SEX_MAPPING = {"male": 0, "female": 1}
EMBARKED_MAPPING = {"S": 0, "C": 1, "Q": 2}
SALUTATION_MAPPING = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
RARE_SALUTATIONS = {
    "Lady",
    "Countess",
    "Capt",
    "Col",
    "Don",
    "Dr",
    "Major",
    "Rev",
    "Sir",
    "Jonkheer",
    "Dona",
}
TICKET_LETTER_MAPPING = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "A": 3,
    "C": 3,
    "P": 0,
    "S": 3,
}
CABIN_LETTER_MAPPING = {"0": 0, "A": 1, "B": 2, "C": 1, "D": 2, "E": 2, "F": 1}


def preprocess(data: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    """Convert raw Titanic rows into numeric model features.

    The function returns a new dataframe and does not mutate the caller's input.
    `train` is used only as the source of imputation statistics.
    """
    processed = data.copy()
    training_reference = train.copy()

    _encode_categoricals(processed)
    _encode_categoricals(training_reference)
    _fill_missing_values(processed, training_reference)
    _add_salutation(processed)
    _add_ticket_features(processed)
    _add_cabin_features(processed)
    _add_family_features(processed)

    return processed


def _encode_categoricals(data: pd.DataFrame) -> None:
    data["Sex"] = data["Sex"].map(SEX_MAPPING).astype("int64")
    data["Embarked"] = data["Embarked"].map(EMBARKED_MAPPING)


def _fill_missing_values(data: pd.DataFrame, train: pd.DataFrame) -> None:
    data["Age"] = data["Age"].fillna(train["Age"].mean())
    data["Embarked"] = data["Embarked"].fillna(train["Embarked"].mode().iloc[0])
    data["Fare"] = data["Fare"].fillna(train["Fare"].mean())


def _add_salutation(data: pd.DataFrame) -> None:
    salutations = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    salutations = salutations.replace(list(RARE_SALUTATIONS), "Rare")
    salutations = salutations.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    data["Salutation"] = salutations.map(SALUTATION_MAPPING).fillna(0).astype("int64")
    data.drop(columns=["Name"], inplace=True)


def _add_ticket_features(data: pd.DataFrame) -> None:
    ticket = data["Ticket"].astype(str)
    ticket_letters = ticket.str[0]
    ticket_letters = np.where(
        ticket_letters.isin(["1", "2", "3", "S", "P", "C", "A"]),
        ticket_letters,
        "0",
    )
    data["Ticket_Lett"] = pd.Series(ticket_letters, index=data.index).map(
        TICKET_LETTER_MAPPING
    )
    data["Ticket_Len"] = ticket.str.len()
    data.drop(columns=["Ticket"], inplace=True)


def _add_cabin_features(data: pd.DataFrame) -> None:
    cabin_letters = data["Cabin"].astype(str).str[0]
    cabin_letters = np.where(
        cabin_letters.isin(["F", "E", "D", "C", "B", "A"]),
        cabin_letters,
        "0",
    )
    data["Cabin_Lett"] = pd.Series(cabin_letters, index=data.index).map(
        CABIN_LETTER_MAPPING
    )
    data.drop(columns=["Cabin"], inplace=True)


def _add_family_features(data: pd.DataFrame) -> None:
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype("int64")
