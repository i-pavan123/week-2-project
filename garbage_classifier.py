import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class GarbageClassifier:
    def __init__(self, img_height=224, img_width=224):
        self.img_height = img_height
        self.img_width = img_width
        self.classes = ['paper', 'metal', 'trash', 'cardboard', 'glass', 'plastic']
        self.num_classes = len(self.classes)
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def load_and_preprocess_data(self, data_path):
        """
        Load and preprocess garbage classification dataset
        Expected folder structure:
        data_path/
        ├── paper/
        ├── metal/
        ├── trash/
        ├── cardboard/
        ├── glass/
        └── plastic/
        """
        print("Loading and preprocessing data...")
        
        images = []
        labels = []
        
        for class_name in self.classes:
            class_path = os.path.join(data_path, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        # Load and preprocess image
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (self.img_width, self.img_height))
                        img = img.astype('float32') / 255.0
                        
                        images.append(img)
                        labels.append(class_name)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = keras.utils.to_categorical(y_encoded, self.num_classes)
        
        print(f"Dataset loaded: {len(X)} images")
        print(f"Classes: {self.classes}")
        
        return X, y_categorical, y
    
    def create_data_generators(self, X_train, y_train, X_val=None, y_val=None, batch_size=32):
        """Create data generators with augmentation for training"""
        
        # Data augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            fill_mode='nearest'
        )
        
        # No augmentation for validation
        val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
        
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        
        if X_val is not None and y_val is not None:
            val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
            return train_generator, val_generator
        
        return train_generator
    
    def build_model(self, model_type='cnn'):
        """Build the classification model"""
        
        if model_type == 'cnn':
            model = keras.Sequential([
                # First Convolutional Block
                layers.Conv2D(32, (3, 3), activation='relu', 
                             input_shape=(self.img_height, self.img_width, 3)),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Second Convolutional Block
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Third Convolutional Block
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Fourth Convolutional Block
                layers.Conv2D(256, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Dense Layers
                layers.GlobalAveragePooling2D(),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
        elif model_type == 'transfer_learning':
            # Using pre-trained MobileNetV2
            base_model = keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
            base_model.trainable = False
            
            model = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function"""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        print(self.model.summary())
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model"""
        
        # Create data generators
        train_gen, val_gen = self.create_data_generators(
            X_train, y_train, X_val, y_val, batch_size
        )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_garbage_classifier.h5', 
                monitor='val_accuracy', 
                save_best_only=True
            )
        ]
        
        # Train the model
        history = self.model.fit(
            train_gen,
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=len(X_val) // batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        test_loss, test_accuracy, test_top3_acc = self.model.evaluate(X_test, y_test, verbose=0)
        
        print(f"\nTest Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Top-3 Accuracy: {test_top3_acc:.4f}")
        
        return test_loss, test_accuracy, test_top3_acc
    
    def predict_single_image(self, image_path):
        """Predict class for a single image"""
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_width, self.img_height))
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Make prediction
            predictions = self.model.predict(img)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            predicted_class = self.classes[predicted_class_idx]
            
            return predicted_class, confidence, predictions[0]
            
        except Exception as e:
            print(f"Error predicting image: {e}")
            return None, None, None
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

# Example usage and training pipeline
def main():
    # Initialize the classifier
    classifier = GarbageClassifier(img_height=224, img_width=224)
    
    # Load and preprocess data
    # Replace 'path/to/your/dataset' with your actual dataset path
    data_path = 'path/to/your/dataset'
    
    try:
        X, y, y_labels = classifier.load_and_preprocess_data(data_path)
        
        # Split the data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        print(f"Training set: {len(X_train)} images")
        print(f"Validation set: {len(X_val)} images")
        print(f"Test set: {len(X_test)} images")
        
        # Build and compile model
        classifier.build_model(model_type='cnn')  # or 'transfer_learning'
        classifier.compile_model(learning_rate=0.001)
        
        # Train the model
        history = classifier.train_model(
            X_train, y_train, X_val, y_val, 
            epochs=50, batch_size=32
        )
        
        # Evaluate the model
        classifier.evaluate_model(X_test, y_test)
        
        # Plot training history
        classifier.plot_training_history(history)
        
        # Save the model
        classifier.save_model('garbage_classifier_final.h5')
        
        # Example prediction
        # predicted_class, confidence, all_probs = classifier.predict_single_image('path/to/test/image.jpg')
        # print(f"Predicted class: {predicted_class} (confidence: {confidence:.4f})")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        print("Please ensure your dataset is organized in the correct folder structure:")
        print("dataset/")
        print("├── paper/")
        print("├── metal/")
        print("├── trash/")
        print("├── cardboard/")
        print("├── glass/")
        print("└── plastic/")

if __name__ == "__main__":
    main()

# Additional utility functions for data analysis
def analyze_dataset(data_path, classes):
    """Analyze dataset distribution"""
    class_counts = {}
    
    for class_name in classes:
        class_path = os.path.join(data_path, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            class_counts[class_name] = count
    
    # Create DataFrame for analysis
    df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
    df['Percentage'] = (df['Count'] / df['Count'].sum()) * 100
    
    print("\nDataset Distribution:")
    print(df)
    
    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.bar(df['Class'], df['Count'])
    plt.title('Garbage Classification Dataset Distribution')
    plt.xlabel('Garbage Type')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return df