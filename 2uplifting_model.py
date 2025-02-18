import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def load_data(filepath: str) -> pd.DataFrame:
    """Load the dataset from a CSV file and print its columns."""
    data = pd.read_csv(filepath)
    print("Columns:", data.columns.tolist())
    return data


def train_model(data: pd.DataFrame, features: list, target: str):
    """
    Trains a single RandomForestClassifier on the dataset.

    Parameters:
        data (pd.DataFrame): The complete dataset.
        features (list): The list of feature columns.
        target (str): The target variable.

    Returns:
        Trained RandomForest model.
    """
    X = data[features]
    y = data[target]

    # Split data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.35, random_state=21
    )

    # Train a RandomForest classifier
    model = RandomForestClassifier(n_estimators=120, random_state=21)
    model.fit(X_train, y_train)

    return model, X_valid


def compute_uplift(model, X_valid: pd.DataFrame) -> pd.DataFrame:
    """
    Computes uplift scores using a single trained model by modifying the coupon variable.

    Parameters:
        model: Trained RandomForest model.
        X_valid (pd.DataFrame): Validation dataset.

    Returns:
        pd.DataFrame: Uplift scores with predictions.
    """
    uplift_df = X_valid.copy()

    # Simulate treatment group (coupon = 1)
    uplift_df["coupon"] = 1
    pred_treatment = model.predict_proba(uplift_df)[:, 1]

    # Simulate control group (coupon = 0)
    uplift_df["coupon"] = 0
    pred_control = model.predict_proba(uplift_df)[:, 1]

    # Compute uplift
    uplift_result_df = pd.DataFrame(
        {
            "probYesCoupon": pred_treatment,
            "probNoCoupon": pred_control,
            "uplift": pred_treatment - pred_control,
        },
        index=X_valid.index,
    )

    return uplift_result_df


def get_top_uplift_records(
    data: pd.DataFrame, top_percent: float = 0.01
) -> pd.DataFrame:
    """
    Filter records with positive uplift and return the top specified percentage.

    Parameters:
        data (pd.DataFrame): The dataset with computed uplift.
        top_percent (float): The top percentage to select (default is 1%).

    Returns:
        pd.DataFrame: The top records based on uplift.
    """
    positive_uplift = data[data["uplift"] > 0].copy()
    positive_uplift.sort_values(by="uplift", ascending=False, inplace=True)
    n_top = max(
        1, int(top_percent * len(positive_uplift))
    )  # Ensure at least one record is returned
    return positive_uplift.head(n_top)


def main():
    filepath = "data/OnlineShopEmailCampaign.csv"
    features = [
        "X1",
        "X2",
        "X3",
        "membership_level_0",
        "membership_level_1",
        "membership_level_2",
        "coupon",
    ]
    target = "conversion"

    # Load the data
    data = load_data(filepath)

    # Train the model
    model, X_valid = train_model(data, features, target)

    # Compute uplift
    uplift_result_df = compute_uplift(model, X_valid)

    # Retrieve and display the top 1% records with positive uplift
    top_records = get_top_uplift_records(uplift_result_df, top_percent=0.01)
    print("Top 1% records based on positive uplift:")
    print(top_records)


if __name__ == "__main__":
    main()
