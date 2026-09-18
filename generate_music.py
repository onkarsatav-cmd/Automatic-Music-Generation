import pickle
import random
import numpy as np
from music21 import chord, instrument, note, stream
from tensorflow.keras.models import load_model

SEQUENCE_LENGTH = 100
OUTPUT_NOTES = 300


def load_training_data():
    with open("notes.pkl", "rb") as f:
        notes = pickle.load(f)

    pitch_names = sorted(set(notes))
    note_to_int = {item: i for i, item in enumerate(pitch_names)}

    network_input = [
        [note_to_int[item] for item in notes[i:i + SEQUENCE_LENGTH]]
        for i in range(len(notes) - SEQUENCE_LENGTH)
    ]

    if not network_input:
        raise RuntimeError("Not enough training data.")

    return network_input, pitch_names


def generate_notes(model, network_input, pitch_names):
    int_to_note = {i: item for i, item in enumerate(pitch_names)}
    pattern = list(random.choice(network_input))
    generated = []

    for _ in range(OUTPUT_NOTES):
        prediction_input = np.reshape(pattern, (1, SEQUENCE_LENGTH, 1))
        prediction_input = prediction_input / float(len(pitch_names))

        prediction = model.predict(prediction_input, verbose=0)[0]
        index = int(np.argmax(prediction))

        generated.append(int_to_note[index])
        pattern.append(index)
        pattern = pattern[1:]

    return generated


def create_midi(prediction_output):
    output_notes = []
    offset = 0.0

    for pattern in prediction_output:
        if "." in pattern:
            pitches = [int(p) for p in pattern.split(".")]
            new_chord = chord.Chord(pitches)
            new_chord.offset = offset
            new_chord.storedInstrument = instrument.Piano()
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp="output/generated_music.mid")
    print("Generated music saved to output/generated_music.mid")


def main():
    model = load_model("music_model.keras")
    network_input, pitch_names = load_training_data()
    prediction_output = generate_notes(model, network_input, pitch_names)
    create_midi(prediction_output)


if __name__ == "__main__":
    main()
