import os
import re

import assemblyai as aai
import contractions
import pandas as pd
from dotenv import load_dotenv
from pytubefix import YouTube
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

# Suppress warnings
"""warning.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['PYTHONWARNINGS'] = 'ignore'  # Suppress Python warnings"""

video_url = "https://www.youtube.com/watch?v=dN3mv5QiZLY&t=952s"
output_path = "extracted_sentences.csv"
load_dotenv()
API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
aai.settings.api_key = API_KEY
model_path = "./translation_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_path)

#  Step 1 - Download and Transcribe Video
"""Downloads a YouTube video, extracts the audio, 
and transcribes it using AssemblyAI."""


def video_transcription(video_url, output_path):
    """Download and transcribe a YouTube video."""
    print("Downloading and transcribing video...")
    # url input from youtube
    yt = YouTube(video_url)

    # extract only audio
    video = yt.streams.filter(only_audio=True).first()

    # download the file
    out_file = video.download()

    # save the file
    base, ext = os.path.splitext(out_file)
    new_file = base + '.mp3'
    os.rename(out_file, new_file)    
    # Configure transcription with Spanish language
    config = aai.TranscriptionConfig(language_code="es")
    transcriber = aai.Transcriber(config=config)

    # Transcribe the audio
    transcript = transcriber.transcribe(new_file)

    # Check for errors
    if transcript.status == aai.TranscriptStatus.error:
        print(f"Transcription failed: {transcript.error}")
        exit(1)

    # Create lists to store data
    sentences = []
    start_times = []
    end_times = []

    # Extract sentences with their timestamps
    sentence_objects = transcript.get_sentences()
    for sentence in sentence_objects:
        sentences.append(sentence.text)
        start_times.append(sentence.start)  # Start time in milliseconds
        end_times.append(sentence.end)      # End time in milliseconds

    # Convert to DataFrame for easier manipulation
    transcript_df = pd.DataFrame({
        'sentence': sentences,
        'start_time_ms': start_times,
        'end_time_ms': end_times
    })

    # Convert milliseconds to a more readable format
    transcript_df['start_time'] = transcript_df['start_time_ms'].apply(
        lambda ms: (
            f"{int(ms/60000):02d}:"
            f"{int((ms%60000)/1000):02d}."
            f"{int(ms%1000):03d}"
        )
    )
    transcript_df['end_time'] = transcript_df['end_time_ms'].apply(
        lambda ms: (
            f"{int(ms/60000):02d}:"
            f"{int((ms%60000)/1000):02d}."
            f"{int(ms%1000):03d}"
        )
    )

    # Format the DataFrame
    transcript_df = transcript_df[['start_time', 'end_time', 'sentence']]

    # Rename the columns as per template
    transcript_df.columns = ['Start Time', 'End Time', 'Sentence']

    # Save to CSV for the next step in your pipeline
    transcript_df.to_csv(output_path, index=False)

    print(
        f"Extracted {len(sentences)} sentences with timestamps "
        f"and saved to {output_path}"
    )
    return transcript_df

# Step 2 - Translate Sentences


def translate(sentence):
    """Translate a given sentence using the pre-trained model."""
    print(f"Translating: {sentence}")
    # Tokenize input sentence
    inputs = tokenizer.encode(
        sentence,
        return_tensors="tf",
        padding=True,
        truncation=True,
        max_length=256
    )
    # Generate translation
    outputs = model.generate(inputs, max_length=256)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Decode the output
    print(f"Translated: {result}")
    return result

# Step 3 - Predict Emotions


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9'\[\].,!?]", " ", text)
    text = ''.join([i for i in text if not i.isdigit()])
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'(?<=\S)[.,](?=\S)', '', text)
    text = " ".join(text.split())
    text = re.sub(r'([.!?])\1+', r'\1', text)
    text = contractions.fix(text)
    return text


def data_processing(df):
    print("Processing data for emotion classification...")

    # Clean the translation column for emotion classification
    df['cleaned_text'] = df['Translation'].astype(str).apply(clean_text)

    # Drop short or empty sentences
    df = df[df['cleaned_text'].str.strip() != '']
    df['text_length'] = df['cleaned_text'].apply(lambda x: len(x.split()))
    df = df[df['text_length'] >= 3]
    df.drop(columns=['text_length'], inplace=True)

    #  Tokenization (assuming you need to recreate or load tokenizer) 
    tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['cleaned_text'])
    sequences = tokenizer.texts_to_sequences(df['cleaned_text'])
    padded = pad_sequences(sequences, padding='post', maxlen=300)  # adjust -- Emils model

    #  Load the .h5 Model
    model = load_model("best_robertaV2")

    #  Predict Emotions
    predictions = model.predict(padded)

    #  Map predictions to labels

    emotion_labels = [
        "neutral", "happiness", "sadness",
        "fear", "anger", "disgust", "surprise"
    ]
    predicted_emotions = [
        emotion_labels[pred.argmax()]
        for pred in predictions
    ]

    # Add to DataFrame
    df['Emotion'] = predicted_emotions
    df.drop(columns=['cleaned_text'], inplace=True)

    # Save the final version
    df.to_csv("translated_emotions.csv", index=False)


if __name__ == '__main__':
    # Step 1: Download and transcribe the video
    video_df = video_transcription(video_url, output_path)

    # Step 2: Translate sentences
    video_df['Translation'] = video_df['Sentence'].apply(translate)
    video_df.to_csv("translated_output.csv", index=False)

    # Step 3: Predict emotions
    df = pd.read_csv("translated_output.csv")
    data_processing(df)
    print("Pipeline completed successfully.")
