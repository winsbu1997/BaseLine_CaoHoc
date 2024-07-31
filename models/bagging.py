from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
from evaluate_model import evaluate_model


def train_bagging(X_train, y_train, X_val, y_val, X_test, y_test, save_data):
    base_model = DecisionTreeClassifier(random_state=42)  
    model = BaggingClassifier(base_model, n_estimators=50, random_state=42)  
    model.fit(X_train, y_train)  
    joblib.dump(model, save_data + "/bagging_classifier.joblib")
    val_acc, val_pre, val_rec, val_f1, test_acc, test_pre, test_rec, test_f1 = evaluate_model(model, X_val, y_val, X_test, y_test)
    return model, val_acc, val_pre, val_rec, val_f1, test_acc, test_pre, test_rec, test_f1
