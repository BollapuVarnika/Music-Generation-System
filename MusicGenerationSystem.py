import os
import random
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
from collections import Counter
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop

# Load MIDI files and extract note pitches
def get_notes_from_midi(folder):
    notes = []
    for file in os.listdir(folder):
        if file.endswith(".mid"):
            midi = pretty_midi.PrettyMIDI(os.path.join(folder, file))
            for instrument in midi.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        notes.append(str(note.pitch))
    return notes

# Plot a histogram of note frequency
def plot_note_histogram(notes):
    note_counts = Counter(notes)
    plt.figure(figsize=(12, 6))
    plt.bar(note_counts.keys(), note_counts.values(), color='skyblue')
    plt.title("Note Frequency Histogram")
    plt.xlabel("Note (Pitch)")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Prepare sequences for training
def prepare_sequences(notes, seq_length):
    encoder = LabelEncoder()
    notes_encoded = encoder.fit_transform(notes)
    vocab_size = len(set(notes_encoded))

    X, y = [], []
    for i in range(len(notes_encoded) - seq_length):
        seq_in = notes_encoded[i:i + seq_length]
        seq_out = notes_encoded[i + seq_length]
        X.append(seq_in)
        y.append(seq_out)

    X = np.array(X)
    y = to_categorical(y, num_classes=vocab_size)
    return X, y, encoder, vocab_size

# Build a deep RNN model with 10 hidden layers
def build_model(seq_length, vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=seq_length))

    # 5 RNN layers
    model.add(SimpleRNN(128, return_sequences=True))
    model.add(SimpleRNN(128, return_sequences=True))
    model.add(SimpleRNN(128, return_sequences=True))
    model.add(SimpleRNN(128, return_sequences=True))
    model.add(SimpleRNN(128))

    # 5 Dense layers
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.001), metrics=['accuracy'])
    return model

# Generate new notes from seed
def generate_notes(model, seed_sequence, encoder, seq_length, gen_length=100):
    generated = list(seed_sequence)
    for _ in range(gen_length):
        input_seq = np.array(generated[-seq_length:])
        input_seq = input_seq.reshape(1, -1)
        prediction = model.predict(input_seq, verbose=0)
        index = np.argmax(prediction)
        generated.append(index)
    return encoder.inverse_transform(generated)

# Save notes as a MIDI file
def create_midi_from_notes(predicted_notes, output_file):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    start = 0
    duration = 0.5
    for note in predicted_notes:
        pitch = int(note)
        note_obj = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + duration)
        instrument.notes.append(note_obj)
        start += duration
    midi.instruments.append(instrument)
    midi.write(output_file)

# Plot training history (loss and accuracy)
def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss', color='red')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Accuracy', color='green')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Visualize generated MIDI notes (like a piano roll)
def plot_generated_midi(predicted_notes):
    times = np.arange(len(predicted_notes)) * 0.5  # Each note lasts 0.5s
    pitches = [int(note) for note in predicted_notes]

    plt.figure(figsize=(12, 5))
    plt.scatter(times, pitches, c=pitches, cmap='viridis', s=60)
    plt.title("Generated MIDI Note Sequence")
    plt.xlabel("Time (s)")
    plt.ylabel("MIDI Pitch")
    plt.colorbar(label='Pitch')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Parameters
    seq_length = 50
    # midi_folder = r"C:/Users/ravik/OneDrive/Documents/archive/split_midi/split_midi"  
    midi_folder = r"C:/Users/ravik/Downloads/data"  
    # Load data and show note frequency
    notes = get_notes_from_midi(midi_folder)
    plot_note_histogram(notes)

    # Data prep
    X, y, encoder, vocab_size = prepare_sequences(notes, seq_length)

    # Build & train model
    model = build_model(seq_length, vocab_size)
    history = model.fit(X, y, epochs=50, batch_size=64)
    model.summary()

    # Plot training metrics
    plot_training_history(history)

    # Generate and visualize music
    seed = X[random.randint(0, len(X)-1)]
    generated_notes = generate_notes(model, seed, encoder, seq_length, gen_length=100)
    plot_generated_midi(generated_notes)

    # Save generated sequence as MIDI
    create_midi_from_notes(generated_notes, "generated_music.mid")
    print("✅ Generated MIDI saved as 'generated_music.mid'")
