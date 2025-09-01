# Model card: Transformer based emotion classification

## Model overview

This model is a fine-tuned roberta-large transformer for multi-class emotion classification transcribed from unscripted Spanish media. The six core emotions described by Ekman & Friesen (1971) serve as the basis for both training and evaluation in this project. The model transcribes Spanish media, translates the sentences into English, and predicts seven emotions: the six core emotion categories defined by Ekman and Friesen: happiness, sadness, anger, surprise, fear, and disgust, along with an additional category, neutral. It integrates traditional NLP features with deep learning to improve performance and interpretability.

## Architecture

- Base model: roberta large which is pretrained on general language understanding tasks  
- Additional feature: Lexicon based emotion count vectors from the NRC emotion Lexicon  
- Tokenizer: RobertaTokenizerFast with the max sequence length set to 256  
- Classification head: Fully connected linear layer with softmax activation for 7 emotion classes  
- Loss Function: Cross-entropy  
- Optimizer: AdamW  
- Evaluation: Epoch-based with early stopping on validation loss  

## Purpose

The model is developed to support emotion recognition in spoken media transcripts, specifically unscripted Spanish reality tv. The goal is to aid the Content Intelligence Agency in analyzing content in video media.

## Development context

- Hardware: Trained via Google Colab Pro using an NVIDIA A100 GPU  
- Training: The duration was about 2 hours over 8 epochs  
- Cost: about €30 for 300 computing units  
- Lexicon: NRC Emotion Lexicon v0.92  
- Team: Spanish NLP group  
- Source: The media used is the Spanish unscripted reality tv show “La Isla de las Tentaciones”  
- Frameworks: Hugging Face Transformers, PyTorch, Pandas, Scikit-learn  

## Intended use

The primary application intended for this model is to tag sentence level emotions in Spanish television show transcripts and support editors with emotion driven video segmentation. This helps the Content Intelligence Agency deliver emotion aware tools content analysis, generate metadata, and help with editorial workflows.

There are a few limitations to take into consideration:
- This is not appropriate for diagnostic emotion classification  
- It is not optimized for real-time systems or high-stake decisions  

## Dataset Details

**Training data:**  
We sourced our data from the GoEmotion dataset which contains over 58K manually annotated data from English Reddit comments and from multiple datasets from multiple student groups which was annotated by the Content Intelligence Agency pipeline. The emotion labels were mapped to the predefined six core emotions + neutral.

**Preprocessing:**
- Normalized emotion labels to the six core emotions + neutral  
- Removed unmapped or invalid labels  
- Applied feature augmentation on Lexicon-based emotions  
- Tokenized using RobertaTokenizerFast  

**Test data:**  
Sourced from transcribed sentences from “La isla de las Tentaciones” and annotated by the Content Intelligence Agency using their pipeline.

**Preprocessing:**
- Manually correcting the Content Intelligence Agency pipeline detected emotion  

## Performance metrics and evaluation

[Detailed error analysis summary](https://github.com/BredaUniversityADSAI/2024-25c-fai2-adsai-group-group16/blob/a8b6be3d6a755a3c1c4b32a729f7017f4d562885/Task_8/Error_Analysis.pdf) 

Our best-performing model (RoBERTa-based) achieved an F1-score of 0.51 on our own validation set. The analysis revealed most errors were due to the misclassification of neutral sentences, often confused with happiness or surprise due to polite wording. Short and ambiguous inputs (e.g., “No!”, “Oh well”) also led to misclassifications due to lack of context. A major issue was poor label quality in the test set, which originated from a noisy transcription pipeline. When tested on other groups’ cleaner datasets (same model, same training), F1 scores improved significantly (up to 75%), confirming that data quality is a key factor affecting performance. Future improvements should focus on cleaner annotations and better handling of subtle, context-dependent language.

## Explainability and transparency

[Detailed explainable AI summary](https://github.com/BredaUniversityADSAI/2024-25c-fai2-adsai-group-group16/blob/fdddc0a3b13b23e5de9761c7f630afcafe0526ab/task_9/XAIROBERTADOC.pdf)

To understand how the model makes decisions, we applied three explainability methods: Gradient × Input, Layer-wise Relevance Propagation (LRP), and Perturbation Testing. Gradient × Input highlighted emotionally charged words (e.g., unloved, obnoxious), while LRP provided more consistent insights across tokens, confirming that the model relies on distributed semantic signals. The Perturbation Test showed that confidence dropped gradually when tokens were removed from subtle sentences, but sharply when key emotional words were removed from explicit ones. These findings confirm that the model avoids over-relying on single keywords and instead builds a robust, distributed understanding of emotional content.

## Recommendation for use

- Performs best on emotionally rich and informal language since the model is trained on that  
- Should be deployed in workflows that has manual validation and human feedback  
- Avoid using this in sensitive settings  

## Sustainability Considerations

**Training:**
- Duration: about 2 hours  
- Hardware: NVIDIA A100 GPU (400W)  
- Energy usage: 0.8kWh     
(This was calculated as: (2 hours × 400 W) / 1000)
 
- Cloud cost: Training used around 300 computing units on Google Colab Pro, which totals to €30  

**Inference:**
- Energy per inference: 0.0056 Wh     
(This was calculated as: (400 W × 5 seconds) / (3600 × 100) = 2000 / 360000)

- Total energy usage: 13.0 Wh (2466 × 0.0056)  

## Optimization strategies

- Early stopping in order to prevent overtraining  
- Lexicon features reduces the model’s reliance on deeper transformer attention  
- Balancing the dataset lowers the epoch requirements and overfitting questions  
