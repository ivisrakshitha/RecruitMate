import speech_recognition as sr
import pyaudio
import wave
import os
import time
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure punkt and stopwords are downloaded
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Set up audio recording parameters
chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
fs = 44100  # Sample rate
audio_filename = "complete_recording.wav"


def record_audio(filename):
    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format, channels=channels,
                    rate=fs, input=True, frames_per_buffer=chunk)
    frames = []

    print("Recording started... Press Ctrl+C to stop.")
    try:
        while True:  # Infinite loop to keep recording until interrupted
            data = stream.read(chunk)
            frames.append(data)
    except KeyboardInterrupt:
        print("Recording stopped.")
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save the recorded data as a single WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()
        print("Recording saved as:", filename)


def convert_audio_to_text(filename):
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        r.adjust_for_ambient_noise(source)
        print("Converting audio to text...")
        audio = r.record(source)

    try:
        # Recognize speech using Google API
        result = r.recognize_google(audio)
        with open("test.txt", "w") as f:
            f.write(result)
        print("Text conversion complete, saved to test.txt.")
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))


def process_text():
    # Read and process the transcribed text from the audio
    with open("test.txt", "r") as file:
        data = file.read()

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(data)
    filtered_sentence = [w for w in word_tokens if w.lower() not in stop_words]

    # Save filtered words to final.txt
    with open('final.txt', 'w') as f:
        f.write(" ".join(filtered_sentence))
    print("Processed text saved to final.txt.")


def check_common_words():
    # Read questions from a predefined file
    with open("paper.txt", "r") as file:
        data = file.read()

    # Remove stopwords from the question file
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(data)
    filtered_questions = [
        w for w in word_tokens if w.lower() not in stop_words]

    # Find common words between filtered_sentence and filtered_questions
    with open("final.txt", "r") as file:
        transcribed_text = file.read()
        transcribed_words = set(word_tokenize(transcribed_text))

    common_words = transcribed_words.intersection(filtered_questions)
    print(f'Number of common elements: {len(common_words)}')
    print("Common elements:", common_words)


# Run the audio recording and processing workflow
record_audio(audio_filename)
convert_audio_to_text(audio_filename)
process_text()
check_common_words()
