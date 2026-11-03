import json
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from src.model import build_model


def main():
    os.makedirs("model", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    np.random.seed(42)
    tf.random.set_seed(42)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    print("Training images:", x_train.shape[0])
    print("Testing images:", x_test.shape[0])

    model = build_model()
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=128,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    model.save("model/digit_cnn.keras")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="Training accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Training loss")
    axes[1].plot(history.history["val_loss"], label="Validation loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig("outputs/training_history.png", dpi=150)
    plt.close(figure)

    matrix = confusion_matrix(y_test, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix)
    display.plot(cmap="Blues", values_format="d")
    plt.title("MNIST Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()

    metrics = {
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "epochs": len(history.history["loss"]),
        "training_images": int(x_train.shape[0]),
        "test_images": int(x_test.shape[0])
    }
    with open("outputs/metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    with open("outputs/classification_report.txt", "w", encoding="utf-8") as file:
        file.write(classification_report(y_test, predictions, digits=4))

    print("\nEvaluation complete.")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print("Saved model to model/digit_cnn.keras")
    print("Saved plots and metrics in outputs/")


if __name__ == "__main__":
    main()
