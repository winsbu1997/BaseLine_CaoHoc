from sklearn.neighbors import KNeighborsClassifier
import joblib
from evaluate_model import evaluate_model
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    pre = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    return acc, pre, rec, f1

def train_knn(X_train, y_train, X_val, y_val, X_test, y_test, save_data):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    #joblib.dump(model, save_data + "knn.joblib")
        # Evaluate on the validation set
    val_acc, val_pre, val_rec, val_f1 = evaluate_model(model, X_val, y_val)
    print("KNN Validation - Accuracy:", val_acc, "Precision:", val_pre, "Recall:", val_rec, "F1 Score:", val_f1)
    
    # Time the evaluation on the test set
    start_time = time.time()
    test_acc, test_pre, test_rec, test_f1 = evaluate_model(model, X_test, y_test)
    end_time = time.time()
    test_time = end_time - start_time
    
    print("KNN Test - Accuracy:", test_acc, "Precision:", test_pre, "Recall:", test_rec, "F1 Score:", test_f1)
    print("Testing time: {:.4f} seconds".format(test_time))
    
    return model, val_acc, val_pre, val_rec, val_f1, test_acc, test_pre, test_rec, test_f1, test_time

