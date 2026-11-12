import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split

# Deep Learning libraries (Used ONLY for the CNN)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# ==========================================
# STEP 1: FROM-SCRATCH FEATURE ENGINES
# ==========================================
def extract_lbp_from_scratch(images_2d):
    print("Extracting LBP Textures from scratch (Pure Math/NumPy)...")
    lbp_features = []
    for img in images_2d:
        padded = np.pad(img, 1, mode='edge')
        h, w = img.shape
        lbp_img = np.zeros((h, w), dtype=int)
        center = padded[1:-1, 1:-1]
        
        # Binary Thresholding & Decimal Conversion
        lbp_img += (padded[0:-2, 0:-2] >= center) * 128 
        lbp_img += (padded[0:-2, 1:-1] >= center) * 64  
        lbp_img += (padded[0:-2, 2:]   >= center) * 32  
        lbp_img += (padded[1:-1, 2:]   >= center) * 16  
        lbp_img += (padded[2:,   2:]   >= center) * 8   
        lbp_img += (padded[2:,   1:-1] >= center) * 4   
        lbp_img += (padded[2:,   0:-2] >= center) * 2   
        lbp_img += (padded[1:-1, 0:-2] >= center) * 1   
        
        histograms = []
        for r in range(0, 24, 12):
            for c in range(0, 24, 12):
                block = lbp_img[r:r+12, c:c+12]
                hist, _ = np.histogram(block, bins=256, range=(0, 256))
                hist = hist.astype(float)
                hist /= (hist.sum() + 1e-6)
                histograms.extend(hist)
        lbp_features.append(histograms)
    return np.array(lbp_features)

def extract_regional_moments_from_scratch(images_2d):
    print("Extracting Geometric Moments from scratch (Pure Math/NumPy)...")
    moment_features = []
    y_indices, x_indices = np.mgrid[0:24, 0:24]
    for img in images_2d:
        img_moments = []
        for r in [0, 12]:
            for c in [0, 12]:
                quadrant = img[r:r+12, c:c+12]
                y_quad = y_indices[r:r+12, c:c+12]
                x_quad = x_indices[r:r+12, c:c+12]
                
                m00 = np.sum(quadrant) + 1e-6
                cx = np.sum(x_quad * quadrant) / m00
                cy = np.sum(y_quad * quadrant) / m00
                mu20 = np.sum(((x_quad - cx)**2) * quadrant) / m00
                mu02 = np.sum(((y_quad - cy)**2) * quadrant) / m00
                img_moments.extend([m00, cx, cy, mu20, mu02])
        moment_features.append(img_moments)
    return np.array(moment_features)

def z_score_normalize(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std

def pure_numpy_cutout_augmentation(X_train, y_train):
    print("Applying pure NumPy Cutout Augmentation to double CNN dataset...")
    X_aug = np.copy(X_train)
    for img in X_aug:
        r = np.random.randint(0, 20)
        c = np.random.randint(0, 20)
        img[r:r+4, c:c+4] = 0.0 
    X_combined = np.vstack((X_train, X_aug))
    y_combined = np.hstack((y_train, y_train))
    return X_combined, y_combined

# ==========================================
# STEP 2: FULL DATA PREPARATION PIPELINE
# ==========================================
print("Loading data...")
try:
    data_x = np.loadtxt('x24x24.txt', skiprows=2)
    data_y = np.loadtxt('y24x24.txt', skiprows=2)
    data_z = np.loadtxt('z24x24.txt', skiprows=2)
    dataset = np.vstack((data_x, data_y, data_z))
    X_raw = dataset[:, :576] 
    y = dataset[:, 578]  
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

X_normalized = X_raw / 255.0 
X_2D = X_normalized.reshape(-1, 24, 24)

X_lbp = extract_lbp_from_scratch(X_2D)
X_moments = extract_regional_moments_from_scratch(X_2D)

# Prevent Data Leakage
X_train_raw, X_test_raw, y_train, y_test, X_train_lbp, X_test_lbp, X_train_mom, X_test_mom = train_test_split(
    X_normalized, y, X_lbp, X_moments, test_size=0.20, stratify=y, random_state=42
)

# Scale moments to prevent RF math distortion
X_train_mom_scaled, X_test_mom_scaled = z_score_normalize(X_train_mom, X_test_mom)

# Combine for the Random Forest
X_train_rf = np.hstack((X_train_raw, X_train_lbp, X_train_mom_scaled))
X_test_rf = np.hstack((X_test_raw, X_test_lbp, X_test_mom_scaled))

# Double the data for the CNN
X_train_2D = X_train_raw.reshape(-1, 24, 24)
X_test_2D = X_test_raw.reshape(-1, 24, 24)
X_train_cnn_aug, y_train_aug = pure_numpy_cutout_augmentation(X_train_2D, y_train)
print(f"Original CNN training samples: {len(X_train_2D)}, After Augmentation: {len(X_train_cnn_aug)}")


X_train_cnn = np.expand_dims(X_train_cnn_aug, -1)
X_test_cnn = np.expand_dims(X_test_2D, -1)

y_train_cat = to_categorical(y_train_aug, num_classes=48)
y_test_cat = to_categorical(y_test, num_classes=48)

# ==========================================
# STEP 3: MODEL A - OPTIMIZED RANDOM FOREST
# ==========================================
print("\nTraining Model A: Random Forest (Pixels + LBP + Scaled Moments)...")
rf_model = RandomForestClassifier(
    n_estimators=150,          
    max_depth=30,              
    min_samples_split=4,       
    max_features='sqrt',       
    n_jobs=-1,                 # Multi-threading for speed!
    random_state=42
)
rf_model.fit(X_train_rf, y_train)
rf_accuracy = rf_model.score(X_test_rf, y_test)

# ==========================================
# STEP 4: MODEL B - REGULARIZED CNN
# ==========================================
print("\nTraining Model B: CNN with Regularization and Callbacks...")
cnn_model = Sequential([
    Input(shape=(24, 24, 1)),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)), 
    Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)), 
    GlobalAveragePooling2D(),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.4),
    Dense(48, activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Speed and Accuracy callbacks
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
early_stopper = EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1)

history = cnn_model.fit(X_train_cnn, y_train_cat, epochs=50, batch_size=32, 
                        validation_data=(X_test_cnn, y_test_cat), 
                        callbacks=[lr_reduction, early_stopper], verbose=1)

# ==========================================
# STEP 5: SMART CONFIDENCE THRESHOLD HYBRID
# ==========================================
print("\nCalculating metrics and engineering Smart Hybrid Predictions...")
rf_probs = rf_model.predict_proba(X_test_rf)
cnn_probs = cnn_model.predict(X_test_cnn, verbose=0)

hybrid_predictions = []
for i in range(len(y_test)):
    rf_confidence = np.max(rf_probs[i])
    cnn_confidence = np.max(cnn_probs[i])
    
    if rf_confidence > 0.40:
        hybrid_predictions.append(np.argmax(rf_probs[i]))
    elif cnn_confidence > 0.40:
        hybrid_predictions.append(np.argmax(cnn_probs[i]))
    else:
        combined = (rf_probs[i] * 0.65) + (cnn_probs[i] * 0.35)
        hybrid_predictions.append(np.argmax(combined))

hybrid_predictions = np.array(hybrid_predictions)
hybrid_accuracy = accuracy_score(y_test, hybrid_predictions)

print(f"\n--- FINAL STABLE SCOREBOARD ---")
print(f"Model A (Random Forest) Accuracy : {rf_accuracy * 100:.2f}%")
print(f"Model B (CNN) Accuracy           : {cnn_model.evaluate(X_test_cnn, y_test_cat, verbose=0)[1] * 100:.2f}%")
print(f"True HYBRID Model Accuracy       : {hybrid_accuracy * 100:.2f}%\n")


# ==========================================
# STEP 6: GENERATE & SAVE REPORT VISUALIZATIONS
# ==========================================
print("\nGenerating and saving graphs for your report...")

# Force Python to use a standard window popup backend
try:
    plt.switch_backend('TkAgg')
except Exception:
    pass

# Ensure interactive plotting is on
plt.ion()

# 1. Plot CNN Loss and Accuracy History (2-Panel Graph)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss Curve
ax1.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
ax1.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
ax1.set_title('CNN Learning Curve (Loss / Error)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epochs', fontsize=10)
ax1.set_ylabel('Loss Value', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=10)

# Accuracy Curve
ax2.plot(history.history['accuracy'], label='Training Accuracy', color='green', linewidth=2)
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red', linewidth=2)
ax2.set_title('CNN Accuracy Trajectory', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epochs', fontsize=10)
ax2.set_ylabel('Accuracy (%)', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig('cnn_learning_curves.png', dpi=300) # HARD COPIED TO YOUR FOLDER!
plt.show()
plt.pause(2.0) # Force the window to stay alive long enough to render

# 2. Plot True Hybrid Confusion Matrix
cm_hybrid = confusion_matrix(y_test, hybrid_predictions)
fig_cm, ax_cm = plt.subplots(figsize=(12, 10))
disp_hybrid = ConfusionMatrixDisplay(confusion_matrix=cm_hybrid)

# Render using the Greens colormap
disp_hybrid.plot(cmap=plt.cm.Greens, ax=ax_cm, xticks_rotation='vertical') 
plt.title("Confusion Matrix (Final True Hybrid Ensemble)", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('hybrid_confusion_matrix.png', dpi=300) # HARD COPIED TO YOUR FOLDER!
plt.show()

# Block execution at the very end so the windows don't instantly close when the script finishes
print("\n[SUCCESS] Script finished! Check your project folder for 'cnn_learning_curves.png' and 'hybrid_confusion_matrix.png'.")
plt.show(block=True)