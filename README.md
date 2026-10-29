# Visionyx

**See beyond the pixels.**

Visionyx is a dark-themed Streamlit image classifier. Upload an image and the app uses TensorFlow/Keras **MobileNetV2**, pretrained on ImageNet, to return its three most likely labels and their prediction scores.

## Features

- Minimal dark interface with electric-blue accents
- Image upload and preview
- Top-three ImageNet predictions with confidence scores
- MobileNetV2 model cached between Streamlit reruns
- RGB image preparation and resizing to the model's expected 224 × 224 input

## Tech stack

- Python
- Streamlit
- TensorFlow / Keras
- MobileNetV2 with ImageNet weights
- Pillow (PIL) for image loading
- NumPy and OpenCV for image preprocessing

## Requirements

- Windows 10 or 11
- Python 3.10 or 3.11 recommended
- Internet access the first time the app loads the pretrained weights

TensorFlow support depends on your Python version and platform. If installation fails, use a Python version supported by the TensorFlow release you intend to install.

## Setup on Windows

Open PowerShell in the project folder.

### Option A: Python virtual environment

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install streamlit tensorflow numpy pillow opencv-python
```

If you have a `requirements.txt`, install the pinned project dependencies instead with `pip install -r requirements.txt`.

### Option B: uv

If [uv](https://docs.astral.sh/uv/) is installed, it can create and manage the virtual environment:

```powershell
uv venv --python 3.11
uv pip install --python .venv\Scripts\python.exe streamlit tensorflow numpy pillow opencv-python
```

To install from a project `pyproject.toml` instead, use `uv sync`.

## Run

From the project folder, with the environment active (or use `uv run`):

```powershell
streamlit run app.py
```

The command prints a local address, usually `http://localhost:8501`, where Visionyx is available in your browser. If your main Python file has a different name, replace `app.py` with that filename.

## Project structure

The following is an **illustrative structure**, not a claim about the files currently present:

```text
visionyx/
├── app.py                   # Streamlit interface and classification flow
├── requirements.txt         # Optional dependency list
├── README.md
└── .streamlit/
    └── config.toml          # Optional Streamlit theme settings
```

## How it works

1. Streamlit accepts an uploaded image and displays a preview.
2. Pillow opens the image; preprocessing converts it to RGB, resizes it to 224 × 224, applies MobileNetV2's `preprocess_input`, and adds a batch dimension.
3. The cached MobileNetV2 model predicts ImageNet class scores.
4. Keras `decode_predictions` turns the highest scores into readable labels, and the app displays the top three.

## Supported image formats

The code shared in the conversation configured the uploader for **JPG/JPEG and PNG**. Image content is converted to RGB for model input, including grayscale or transparent images if the uploader accepts them. Add other extensions to Streamlit's uploader configuration before expecting formats such as WebP or GIF to work.

## Model limitations

- MobileNetV2 recognizes the 1,000 ImageNet categories; it cannot identify arbitrary objects or custom categories unless the model is retrained or replaced.
- The top result is a model prediction, not a guaranteed identification. Scores are not necessarily calibrated probabilities.
- Cropping, small objects, unusual viewpoints, poor lighting, and images unlike ImageNet training data can reduce accuracy.
- The standard model processes one 224 × 224 image at a time, so fine details may be lost during resizing.
- The pretrained weights may need to download on first use, and TensorFlow can require substantial memory and disk space.

## Troubleshooting

| Problem | What to try |
| --- | --- |
| `streamlit` is not recognized | Activate `.venv`, then run `python -m streamlit run app.py`. |
| PowerShell blocks environment activation | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that PowerShell window, then activate again. |
| TensorFlow installation fails | Confirm the Python version and architecture are supported by the TensorFlow release; Python 3.11 64-bit is a reasonable starting point. |
| The app cannot download ImageNet weights | Check the network connection and proxy/firewall settings, then restart the app. |
| Image preprocessing or prediction errors | Confirm the uploaded file is a valid image, convert it to RGB before creating the NumPy array, and return the processed array from the preprocessing function. |
| Model loads repeatedly or slowly | Put model construction in a function decorated with `@st.cache_resource` and call that cached function. |
| `cv2` cannot be imported | Install OpenCV into the active environment with `pip install opencv-python` (or the equivalent `uv pip install`). |

> **Implementation note:** In the original code shared in chat, `preprocess_image()` did not return its processed array, and the cached model function was defined but not called. The classification flow needs those two fixes to work as described above. The function was also named `calssify_image`; correcting it to `classify_image` is recommended for clarity.

Please assume that this README was generated by AI.
