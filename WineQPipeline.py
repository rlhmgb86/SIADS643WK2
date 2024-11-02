import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from ucimlrepo import fetch_ucirepo


class WineQualityPipeline:
    """Wine quality prediction pipeline."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        """Initialize pipeline.

        Args:
            test_size (float, optional): Test size. Defaults to 0.2.
            random_state (int, optional): Random state. Defaults to 42.
        """
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self) -> pd.DataFrame:
        """Load wine quality data.

        Returns:
            pd.DataFrame: Loaded data.
        """
        wine = fetch_ucirepo(id=109)
        df = wine.data.features.copy()
        df['Quality'] = wine.data.targets
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data.

        Args:
            df (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: Preprocessed data.
        """
        features = ['Alcohol', 'Malicacid', 'Ash', 'Magnesium', 'Total_phenols', 'Flavanoids', 'Proline']
        df = df[features + ['Quality']]
        return df

    def split_data(self, df: pd.DataFrame) -> tuple:
        """Split data into training and testing sets.

        Args:
            df (pd.DataFrame): Input data.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        X = df.drop('Quality', axis=1)
        y = df['Quality']
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
        """Train Linear Regression and Random Forest models.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training target.

        Returns:
            tuple: Trained models.
        """
        lr_model = LinearRegression()
        rf_model = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
        lr_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)
        return lr_model, rf_model

    def evaluate_models(self, lr_model, rf_model, X_test: pd.DataFrame) -> tuple:
        """Evaluate trained models.

        Args:
            lr_model: Linear Regression model.
            rf_model: Random Forest model.
            X_test (pd.DataFrame): Testing features.

        Returns:
            tuple: Predictions.
        """
        lr_y_pred = lr_model.predict(X_test)
        rf_y_pred = rf_model.predict(X_test)
        return lr_y_pred, rf_y_pred

    def calculate_metrics(self, y_test: pd.Series, lr_y_pred: pd.Series, rf_y_pred: pd.Series) -> dict:
        """Calculate evaluation metrics.

        Args:
            y_test (pd.Series): Testing target.
            lr_y_pred (pd.Series): Linear Regression predictions.
            rf_y_pred (pd.Series): Random Forest predictions.

        Returns:
            dict: Evaluation metrics.
        """
        metrics = {
            'Linear Regression': {
                'MSE': mean_squared_error(y_test, lr_y_pred),
                'R²': r2_score(y_test, lr_y_pred)
            },
            'Random Forest': {
                'MSE': mean_squared_error(y_test, rf_y_pred),
                'R²': r2_score(y_test, rf_y_pred)
            }
        }
        return metrics

    def run_pipeline(self) -> None:
        """Run the wine quality prediction pipeline."""
        df = self.load_data()
        df = self.preprocess_data(df)
        X_train, X_test, y_train, y_test = self.split_data(df)
        lr_model, rf_model = self.train_models(X_train, y_train)
        lr_y_pred, rf_y_pred = self.evaluate_models(lr_model, rf_model, X_test)
        metrics = self.calculate_metrics(y_test, lr_y_pred, rf_y_pred)
        for model, results in metrics.items():
            print(f"\n{model}:")
            for metric, value in results.items():
                print(f"{metric}: {value:.2f}")


if __name__ == "__main__":
    pipeline = WineQualityPipeline()
    pipeline.run_pipeline()