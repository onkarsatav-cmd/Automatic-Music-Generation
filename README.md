# Automatic Music Generation Using LSTM

A deep learning project that learns musical patterns from MIDI files and generates new music using an LSTM (Long Short-Term Memory) neural network.

## Project Overview

The system:

1. Reads MIDI files from the dataset.
2. Extracts notes and chords using Music21.
3. Converts musical sequences into numerical data.
4. Trains an LSTM neural network.
5. Predicts the next musical notes.
6. Creates a new MIDI file.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Music21
- LSTM
- Deep Learning

## Project Structure

```text
automatic-music-generation/
├── data/
│   └── midi/
│       └── .gitkeep
├── output/
│   └── .gitkeep
├── train_model.py
├── generate_music.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

Add legally usable MIDI files to:

```text
data/midi/
```

Use MIDI files that you created yourself or that are public-domain / appropriately licensed.

## Train the Model

```bash
python train_model.py
```

Training creates:

- `music_model.keras`
- `notes.pkl`

These generated training artifacts are ignored by Git.

## Generate Music

After training:

```bash
python generate_music.py
```

The generated MIDI file will be saved as:

```text
output/generated_music.mid
```

You can open the MIDI file in MuseScore or another MIDI-compatible player.

## Machine Learning Workflow

```text
MIDI Dataset
     ↓
Note & Chord Extraction
     ↓
Sequence Preparation
     ↓
LSTM Neural Network
     ↓
Model Training
     ↓
Next-Note Prediction
     ↓
Generated MIDI Music
```

## Model Architecture

- LSTM: 256 units
- Dropout: 0.3
- LSTM: 256 units
- Dropout: 0.3
- LSTM: 256 units
- Dense: 256 units
- Dropout: 0.3
- Output: Softmax

## Future Improvements

- Temperature-based sampling
- Multiple instruments
- Genre-specific models
- Larger MIDI datasets
- Web interface with Streamlit
- Better music evaluation metrics

## Author

**Your Name**

GitHub: `https://github.com/yourusername`
