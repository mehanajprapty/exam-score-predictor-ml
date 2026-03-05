import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def create_dataset():
    """
    Creates training dataset.
    X contains [study_hours, sleep_hours]
    y contains exam scores.
    """
    X = np.array([
        [1, 7],
        [2, 6],
        [3, 8],
        [4, 5],
        [5, 6],
        [6, 4],
        [7, 7],
        [8, 3]
    ])

    y = np.array([38, 44, 52, 53, 61, 63, 74, 72])

    return X, y


def train_model(X, y):
    """
    Trains Linear Regression model.
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def evaluate_model(model, X, y):
    """
    Evaluates model using R² score.
    """
    predictions = model.predict(X)
    score = r2_score(y, predictions)
    return score


def main():
    print("\n=== Exam Score Prediction Model ===\n")

    # Create dataset
    X, y = create_dataset()

    # Train model
    model = train_model(X, y)

    # Evaluate model
    r2 = evaluate_model(model, X, y)

    print("Model trained successfully!\n")
    print(f"Weight for Study Hours: {model.coef_[0]:.2f}")
    print(f"Weight for Sleep Hours: {model.coef_[1]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"R² Score: {r2:.4f}")

    # User input
    study = float(input("\nEnter study hours: "))
    sleep = float(input("Enter sleep hours: "))

    prediction = model.predict([[study, sleep]])

    print(f"\nPredicted exam score: {prediction[0]:.2f}")


if __name__ == "__main__":
    main()

