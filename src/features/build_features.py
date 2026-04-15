def engineer_features(df):
    df = df.copy()
    
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
    df["Deck"] = df["Cabin"].str[0]
    df["Deck"] = df["Deck"].fillna("Missing")
    
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    
    df = df.drop(columns=["Name", "Cabin", "Ticket", "PassengerId"])
    
    return df