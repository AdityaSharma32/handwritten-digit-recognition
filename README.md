# AI-Based Handwritten Digit Recognition System

A student-friendly CNN project based on the uploaded Major Project brief. It trains on MNIST, evaluates the model, saves plots, and provides a Streamlit interface for uploading a digit image.

## 1. Requirements

- Python 3.10 or 3.11 recommended
- VS Code
- Internet connection for the first training run (TensorFlow downloads MNIST)

## 2. Open the folder in VS Code

1. Extract `handwritten_digit_recognition.zip`.
2. Open VS Code.
3. Select **File → Open Folder** and choose `handwritten_digit_recognition`.
4. Open **Terminal → New Terminal**.

## 3. Create and activate a virtual environment

### Windows (PowerShell)
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If TensorFlow installation fails, check that your Python version is supported by the TensorFlow release available for your operating system. Python 3.10 or 3.11 is a good starting point.

## 5. Train the CNN

```bash
python train.py
```

This downloads MNIST through Keras on the first run, trains the model, and creates:

- `model/digit_cnn.keras` — trained model
- `outputs/training_history.png` — accuracy and loss curves
- `outputs/confusion_matrix.png` — test-set confusion matrix
- `outputs/metrics.json` — actual test accuracy and loss

Training time depends on your computer. The model is not included as a pre-trained file; train it locally so the saved metrics reflect your own run.

## 6. Run the web interface

After training finishes:

```bash
streamlit run app.py
```

Streamlit will show a local URL, usually `http://localhost:8501`. Open it in your browser, upload an image containing one handwritten digit, and view the prediction and confidence score.

## 7. Preparing a test image

For best results, upload a clear image containing one digit only. Use a plain background, keep the digit centered, and avoid extra text. The app converts the image to grayscale, determines whether to invert it, resizes it to 28×28, and normalizes the pixels.

The MNIST model was trained on small, centered, grayscale digit images. Photos with shadows, colored backgrounds, multiple digits, or unusual handwriting may reduce accuracy.

## 8. Project files

- `train.py` — downloads MNIST, builds and trains the CNN, evaluates it, and saves plots and metrics.
- `app.py` — Streamlit upload and prediction interface.
- `src/model.py` — CNN architecture.
- `src/preprocessing.py` — image preprocessing for uploaded images.
- `requirements.txt` — Python dependencies.
- `report_template.md` — report outline. Fill in the values from your actual run.
- `PROJECT_PLAN.md` — implementation plan and checklist.

## 9. Troubleshooting

- **`ModuleNotFoundError`**: activate `.venv` and run `pip install -r requirements.txt`.
- **Model file not found**: run `python train.py` successfully before starting Streamlit.
- **TensorFlow installation error**: use Python 3.10/3.11 and a compatible TensorFlow build; avoid installing packages into a different Python interpreter.
- **Poor predictions on photos**: try a clean, high-contrast image with one centered digit. This model is trained on MNIST, not on arbitrary camera photos.
- **Port 8501 is busy**: run `streamlit run app.py --server.port 8502`.

## 10. Before submission

1. Run training and save the generated metrics and plots.
2. Test the app with several images of your own.
3. Add screenshots of the running app and terminal to your report.
4. Complete `report_template.md` using your actual outputs.
5. Make sure you can explain convolution, pooling, normalization, epochs, accuracy, loss, and the confusion matrix in your own words.

Do not report an accuracy value until you have run the test evaluation.
