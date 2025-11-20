# 🏆 Grammar and Speech Scoring Engine

An advanced machine learning pipeline designed to predict a score (e.g., grading an oral submission) based on an individual's speech, combining both linguistic and acoustic quality features extracted from audio recordings.

-----

##  Features

This project leverages state-of-the-art models and comprehensive feature engineering to provide a robust scoring prediction:

  * **Automatic Speech Recognition (ASR):** Uses the **Whisper** model from Hugging Face `transformers` for high-accuracy audio transcription, essential for the linguistic analysis.
  * **Comprehensive Feature Engineering:** Extracts over 60+ features across three main domains:
      * **Linguistic & Grammar:** Features like grammar and typo counts using `language-tool-python`, tense errors, and sentence complexity metrics (e.g., **Syntax Tree Depth** using `spaCy`).
      * **Readability & Vocabulary:** Readability scores (**Flesch**, **Flesch-Kincaid** using `textstat`), lexical diversity (**Type-Token Ratio - TTR**), and unique word count.
      * **Acoustic (Audio):** Speaking rate, audio duration, silence ratio, mean pitch, and pitch range, calculated using `librosa` and `pydub`.
  * **Machine Learning Scoring:** A predictive model based on a **CatBoost** regressor, trained using a **5-Fold Cross-Validation** and a **Meta Ensemble** approach to maximize prediction accuracy.

-----

## 🛠️ Installation

The project requires Python, Java, and several libraries.

### Prerequisites

1.  **Python 3.x**
2.  **Java Runtime Environment (JRE) or Java Development Kit (JDK):** The LanguageTool component requires Java. The notebook specifies installing **OpenJDK 17**.

### Python Environment Setup

Install all necessary Python packages.

```bash
# Install core dependencies (Pytorch, HuggingFace Transformers)
!pip install -q transformers torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Install NLP, Audio, ML, and Utility libraries
!pip install -q librosa soundfile language-tool-python pandas numpy scikit-learn catboost textstat sentence-transformers pydub tenacity

# Install Spacy language model
!python -m spacy download en_core_web_sm
```

### Java Setup (if not already installed)

This step is typically required in constrained environments like Google Colab.

```bash
!sudo apt-get install openjdk-17-jdk -y
!sudo update-alternatives --set java /usr/lib/jvm/java-17-openjdk-amd64/bin/java
```

-----

##  Usage

The engine is designed to process audio files, transcribe them, and generate a final predicted score for each.

### 1\. Data Preparation

  * Place all audio files (`.wav` format is preferred) into a dedicated input directory (e.g., `test_audio_dir`).
  * Ensure a corresponding CSV file (`test.csv`) is available, containing the list of audio filenames for submission.

### 2\. Prediction Pipeline

The core logic of the pipeline involves the following steps for each audio file:

1.  **Audio Loading:** The audio file is loaded robustly, first attempting `pydub` and then falling back to `torchaudio` if needed.
2.  **Transcription:** The **Whisper** model transcribes the audio into text.
3.  **Feature Extraction:** Over 60 features are calculated from the text and the raw audio.
4.  **Scoring:** The extracted feature vector is passed to the trained **CatBoost** ensemble model, which outputs the final predicted score.

### 3\. Generating Submission

The final step aggregates all predicted scores into a standard submission format, typically a CSV file with `filename` and the predicted `label` (score).

The script outputs a file named `submission.csv` (or similar, like `submission_TOP5_SHL.csv`).

-----

##  Evaluation (Training/Validation)

During development, the model was evaluated on a validation set (size 82).

The primary metrics used were:

  * **Root Mean Square Error (RMSE):** Measures the average magnitude of the errors. A lower RMSE indicates better performance. The reported validation RMSE was **0.7469**.
  * **Pearson Correlation:** Measures the linear correlation between the true scores and the predicted scores.

Visualizations such as a **Confusion Matrix** and an **Error Boxplot** were generated to analyze model performance across different true score levels.
