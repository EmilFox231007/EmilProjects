# Video Transcription, Translation, and Emotion Detection Pipeline

This project provides a complete pipeline that downloads a YouTube video, extracts its audio, transcribes the audio using AssemblyAI, translates the transcription using a pre-trained translation model (via the Transformers library), and detects emotions in the translated text using a TensorFlow Keras model.

---

## Requirements

Ensure your Python environment includes the following packages (minimum versions specified where applicable):

- **pandas>=1.0.0**
- **assemblyai**
- **deep_translator>=1.8.1**
- **pytube>=11.0.0**
- **tensorflow**
- **tf-keras** 
- **transformers**
- **datasets**
- **ftfy**
- **sentencepiece**
- **torch>=2.0.0**
- **pytubefix**
- **regex**
- **emoji**
- **contractions**
- **sacremoses**

---

## Installation

1. **Clone the Repository**

   ```bash
   gh repo clone BredaUniversityADSAI/2024-25c-fai2-adsai-group-group16
   ```

2. **Create a Virtual Environment (Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   Install all required packages using pip:

   ```bash
   pip install -r requirements.txt
---

## Usage

Before running the pipeline, update the following configurations in your code:

- **AssemblyAI API Key:** To create an assemblyai key you need to access the https://www.assembly.ai website. Then add the key to a `.env` file inside the `task_11` with 
```bash
ASSEMBLYAI_API_KEY=<YOUR_KEY>
```
Or replace the `aai.settings.api_key = API_KEY` on the `task_11/pipeline.py`
- **Translation Model Path:** Ensure the `model_path` variable points to your pre-trained translation model directory.
- **Video URL:** Modify the `video_url` variable if you need to process a different YouTube video.
- **Emotion Classification Model:** Adjust the model path in the emotion detection section (currently using `"tf_model.h5"`).

Run the pipeline by executing the script:

```bash
python task_11/pipeline.py
```

The pipeline will execute the following steps:

1. **Download & Transcription:**
   - Downloads the YouTube video.
   - Extracts audio and converts it to MP3.
   - Transcribes audio using AssemblyAI.
   - Saves the transcript with timestamps to `extracted_sentences.csv`.

2. **Translation:**
   - Translates each transcribed sentence using the translation model.
   - Saves the translated output to `translated_output.csv`.

3. **Emotion Detection:**
   - Cleans and processes the translated sentences.
   - Tokenizes and pads text.
   - Predicts emotions using the TensorFlow Keras model.
   - Saves the final results to `translated_emotions.csv`.

---

## Code Structure & Functionality

### 1. Video Transcription

- **Function:** `video_transcription(video_url, output_path)`
- **Description:**  
  - Downloads the YouTube video.
  - Extracts audio and converts it to MP3.
  - Transcribes the audio using AssemblyAI.
  - Processes the transcript into a CSV file that includes sentences along with their start and end timestamps.

### 2. Translation

- **Function:** `translate(sentence)`
- **Description:**  
  - Tokenizes an input sentence.
  - Translates the sentence using a pre-trained model from the Transformers library.
  - Outputs the translated text.

### 3. Emotion Detection

- **Functions:**
  - `clean_text(text)`: Cleans and preprocesses the translated text (e.g., lowercasing, regex filtering, contraction expansion).
  - `data_processing(df)`:  
    - Cleans, tokenizes, and pads the text.
    - Loads a pre-trained TensorFlow Keras model.
    - Predicts emotions from the processed text.
    - Saves the output to `translated_emotions.csv`.