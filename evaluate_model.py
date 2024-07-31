from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define a function to evaluate the model
def evaluate_model(model, X_val, y_val, X_test, y_test):
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_pre = precision_score(y_val, y_val_pred, average='binary')
    val_rec = recall_score(y_val, y_val_pred, average='binary')
    val_f1 = f1_score(y_val, y_val_pred, average='binary')
    
    # Test metrics
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_pre = precision_score(y_test, test_pred, average='binary')
    test_rec = recall_score(y_test, test_pred, average='binary')
    test_f1 = f1_score(y_test, test_pred, average='binary')

    return val_acc, val_pre, val_rec, val_f1, test_acc, test_pre, test_rec, test_f1