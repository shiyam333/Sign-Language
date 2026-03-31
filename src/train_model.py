import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load data
df = pd.read_csv("../data/asl_landmarks.csv")  # adjust path if different

# 2. Features and labels
X = df.drop(columns=["label"]).values  # shape: (2203, 63)
y = df["label"].values                 # A-Z, space, del

# 3. Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

num_classes = len(le.classes_)
input_dim = X.shape[1]

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. Build MLP model
model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 6. Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=40,
    batch_size=32
)

# 7. Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Test accuracy:", test_acc)

# 8. Save model and label map
model.save("../models/asl_mlp.h5")
np.save("../models/label_classes.npy", le.classes_)
print("Saved model to ../models/asl_mlp.h5")
print("Saved label classes to ../models/label_classes.npy")
