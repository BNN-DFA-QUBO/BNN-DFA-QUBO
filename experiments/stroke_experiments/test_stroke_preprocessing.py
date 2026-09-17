from utils.stroke_data import prepare_stroke_data


X_train, X_val, X_test, y_train, y_val, y_test = prepare_stroke_data()

print("Train:", X_train.shape, y_train.shape)
print("Validation:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

print("\nStroke distribution:")
print("Train:")
print(y_train.value_counts())

print("\nValidation:")
print(y_val.value_counts())

print("\nTest:")
print(y_test.value_counts())

print("\nNumber of input features:", X_train.shape[1])