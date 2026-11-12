import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical

# ==========================================
# STEP 1: LOAD THE DATA
# ==========================================
# Each file: line 1 = number of images, line 2 = pixels per image (576 = 24x24),
# then one image per line: 576 pixel values followed by 8 attributes.
# Attribute a3 (column 578) is the person ID (0-47).
print("Loading data...")
data = np.vstack([np.loadtxt(f, skiprows=2)
                  for f in ("x24x24.txt", "y24x24.txt", "z24x24.txt")])

X = data[:, :576]               # pixel values (already scaled in the dataset, range ~ -1.5 to 2.8)
y = data[:, 578].astype(int)    # person ID

# No /255 scaling: the pixels are not 0-255, and dividing them
# shrinks the inputs so much that the neural network barely learns.

# Stratified 80/20 split, so every person appears in train and test in the same proportion
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================================
# STEP 2: MODEL A - RANDOM FOREST
# ==========================================
print("\nTraining Model A: Random Forest...")
# We use Random Forest because a Decision Tree is highly prone to overfitting with 48 classes.
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate Random Forest
rf_accuracy = rf_model.score(X_test, y_test)
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")


# ==========================================
# STEP 3: MODEL B - ANN WITH BACKPROPAGATION
# ==========================================
print("\nTraining Model B: ANN with Backpropagation...")
# Convert labels to categorical (one-hot encoding) for the Softmax output
y_train_cat = to_categorical(y_train, num_classes=48)
y_test_cat = to_categorical(y_test, num_classes=48)

# Architecture: Input (576) -> Hidden Layer 1 (128) -> Hidden Layer 2 (64) -> Output (48)
ann_model = Sequential([
    Input(shape=(576,)),                      # Input layer: 576 pixels
    Dense(128, activation='relu'),            # Hidden Layer 1: 128 nodes, ReLU activation
    Dense(64, activation='relu'),             # Hidden Layer 2: 64 nodes, ReLU activation
    Dense(48, activation='softmax')           # Output: 48 nodes, Softmax for multi-class
])

# Compile the model using backpropagation (adam optimizer)
ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model and save the history to plot the Loss Curve later
history = ann_model.fit(X_train, y_train_cat, epochs=30, batch_size=32, validation_data=(X_test, y_test_cat))


# ==========================================
# STEP 4: SUCCESS METRICS
# ==========================================
print("\nGenerating Success Metrics...")

# 1. Confusion Matrix for Random Forest (To see which celebrities the model confuses)
rf_predictions = rf_model.predict(X_test)
cm = confusion_matrix(y_test, rf_predictions)

plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
plt.title("Confusion Matrix (Random Forest)")
plt.savefig("confusion_matrix_rf.png", dpi=150, bbox_inches="tight")
plt.show()

# 2. Loss Curve for ANN (To demonstrate that the model is learning effectively)
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('ANN Loss Curve')
plt.xlabel('Epochs (Training Steps)')
plt.ylabel('Loss (Error)')
plt.legend()
plt.savefig("confusion_matrix_rf.png", dpi=150, bbox_inches="tight")
plt.show()

# ==========================================
# STEP 5: THE TRUE HYBRID (SOFT VOTING)
# ==========================================
print("\nEvaluating True Hybrid Model (Soft Voting)...")

# 1. Get the probability (confidence) scores from both models for the test set
rf_probs = rf_model.predict_proba(X_test) 
ann_probs = ann_model.predict(X_test, verbose=0)

# 2. Average the probabilities together (This is the "Hybrid" part!)
# We weigh them equally (50/50)
hybrid_probs = (rf_probs + ann_probs) / 2.0

# 3. Find the class with the highest average probability
hybrid_predictions = np.argmax(hybrid_probs, axis=1)

# 4. Calculate final accuracy
from sklearn.metrics import accuracy_score
hybrid_accuracy = accuracy_score(y_test, hybrid_predictions)

# 5. Print the Final Scoreboard
print(f"\n--- FINAL SCOREBOARD ---")
print(f"Model A (Random Forest) Accuracy : {rf_accuracy * 100:.2f}%")
ann_loss, ann_acc = ann_model.evaluate(X_test, y_test_cat, verbose=0)
print(f"Model B (ANN) Accuracy           : {ann_acc * 100:.2f}%")
print(f"True HYBRID Model Accuracy       : {hybrid_accuracy * 100:.2f}%\n")

# 6. Show the Hybrid Confusion Matrix (Using Green to distinguish it)
cm_hybrid = confusion_matrix(y_test, hybrid_predictions)
plt.figure(figsize=(10, 8))
disp_hybrid = ConfusionMatrixDisplay(confusion_matrix=cm_hybrid)
disp_hybrid.plot(cmap=plt.cm.Greens, ax=plt.gca()) 
plt.title("Confusion Matrix (True Hybrid Model)")
plt.savefig("confusion_matrix_rf.png", dpi=150, bbox_inches="tight")
plt.show()
