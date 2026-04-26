import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical

# ==========================================
# STEP 1: LOAD AND NORMALIZE THE DATA
# ==========================================
print("Loading data...")
# NOTE: We assume the data files contain pixel values and the last column is the person's ID (label).
# You may need to check 'Readme.txt' to confirm if the labels are arranged differently!
# ==========================================
# STEP 1: LOAD AND NORMALIZE THE DATA
# ==========================================

try:
    # We use skiprows=2 to ignore the <L> and <N> metadata lines at the top of the files
    data_x = np.loadtxt('x24x24.txt', skiprows=2)
    data_y = np.loadtxt('y24x24.txt', skiprows=2)
    data_z = np.loadtxt('z24x24.txt', skiprows=2)
    
    # Combine the 3 separate files into one big dataset 
    dataset = np.vstack((data_x, data_y, data_z))
    
    # X (Features): The first 576 columns are the pixel intensities
    X = dataset[:, :576] 
    
    # y (Labels): The 'a3' attribute is the person's class number.
    # Since columns are 0-indexed and pixels take up 0-575:
    # a1 = 576, a2 = 577, a3 = 578
    y = dataset[:, 578]  

except Exception as e:
    print(f"Error loading files: {e}")
    exit()

# Action: Normalize the data (scaling it to the [0,1] range) to prevent Gradient Explosion
# The Readme states intensities are grayscale (typically 0-255)
X_normalized = X / 255.0 

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

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
plt.show()

# 2. Loss Curve for ANN (To demonstrate that the model is learning effectively)
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('ANN Loss Curve')
plt.xlabel('Epochs (Training Steps)')
plt.ylabel('Loss (Error)')
plt.legend()
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
plt.show()