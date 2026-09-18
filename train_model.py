import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

SEQUENCE_LENGTH = 100


def extract_notes():
    notes = []

    for file in glob.glob("data/midi/*.mid"):
        print(f"Reading: {file}")
        try:
            midi = converter.parse(file)
            parts = instrument.partitionByInstrument(midi)
            elements = parts.parts[0].recurse() if parts else midi.flat.notes

            for element in elements:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append(".".join(str(n) for n in element.normalOrder))
        except Exception as exc:
            print(f"Could not read {file}: {exc}")

    if not notes:
        raise RuntimeError("No MIDI notes found. Add .mid files to data/midi/.")

    with open("notes.pkl", "wb") as f:
        pickle.dump(notes, f)

    return notes


def prepare_sequences(notes):
    pitch_names = sorted(set(notes))
    note_to_int = {item: i for i, item in enumerate(pitch_names)}

    network_input = []
    network_output = []

    for i in range(len(notes) - SEQUENCE_LENGTH):
        network_input.append(
            [note_to_int[item] for item in notes[i:i + SEQUENCE_LENGTH]]
        )
        network_output.append(note_to_int[notes[i + SEQUENCE_LENGTH]])

    if not network_input:
        raise RuntimeError(
            f"Not enough notes. Need more than {SEQUENCE_LENGTH} notes."
        )

    network_input = np.reshape(
        network_input, (len(network_input), SEQUENCE_LENGTH, 1)
    )
    network_input = network_input / float(len(pitch_names))
    network_output = to_categorical(network_output, num_classes=len(pitch_names))

    return network_input, network_output, pitch_names


def build_model(network_input, n_vocab):
    model = Sequential([
        LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]),
             return_sequences=True),
        Dropout(0.3),
        LSTM(256, return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(n_vocab, activation="softmax"),
    ])

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    return model


def main():
    notes = extract_notes()
    network_input, network_output, pitch_names = prepare_sequences(notes)

    model = build_model(network_input, len(pitch_names))
    model.summary()

    model.fit(
        network_input,
        network_output,
        epochs=30,
        batch_size=64,
        validation_split=0.1,
        verbose=1,
    )

    model.save("music_model.keras")
    print("Training complete. Model saved as music_model.keras")


if __name__ == "__main__":
    main()
