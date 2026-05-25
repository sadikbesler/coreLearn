from abc import ABC, abstractmethod


class BaseModel(ABC):
    """
    Abstract base class for all models.

    Implements the **Template Method** design pattern: ``fit_predict`` defines
    the skeleton (fit → predict) and subclasses fill in the concrete steps.
    """

    @abstractmethod
    def fit(self, X, y) -> "BaseModel":
        """Train the model on feature matrix X and label vector y."""
        pass

    @abstractmethod
    def predict(self, X) -> list:
        """Return predictions for feature matrix X."""
        pass

    def fit_predict(self, X_train, y_train, X_test) -> list:
        """Template method: train on X_train/y_train, then predict X_test."""
        self.fit(X_train, y_train)
        return self.predict(X_test)
