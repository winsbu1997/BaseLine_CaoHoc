from sklearn.neural_network import MLPClassifier
import joblib
from evaluate_model import evaluate_model

def train_mlp(X_train, y_train, X_val, y_val, X_test, y_test, save_data):
    model = MLPClassifier(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, save_data + "mlp.joblib")
    val_acc, val_pre, val_rec, val_f1, test_acc, test_pre, test_rec, test_f1 = evaluate_model(model, X_val, y_val, X_test, y_test)
    print("MLP Validation - Accuracy:", val_acc, "Precision:", val_pre, "Recall:", val_rec, "F1 Score:", val_f1)
    return model, val_acc, val_pre, val_rec, val_f1, test_acc, test_pre, test_rec, test_f1
