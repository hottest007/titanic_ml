def engineer_features(df):
    df = df.copy()

    if "Name" in df.columns:
        df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")

    if "Cabin" in df.columns:
        df["Deck"] = df["Cabin"].str[0]
        df["Deck"] = df["Deck"].fillna("Missing")

    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    df = df.drop(columns=["Name", "Cabin", "Ticket", "PassengerId"], errors="ignore")

    return df