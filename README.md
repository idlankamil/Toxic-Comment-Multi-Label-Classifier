# 🛡️ Toxic Comment Classifier
A multi-label comment toxicity classifier built with a hybrid NLP pipeline in Python.
Detects six toxicity types simultaneously using a combination of TF-IDF, sentiment features,
and two tuned models — Random Forest and Neural Network — compared head-to-head.

✨ Technologies
* `Python`
* `scikit-learn`
* `TensorFlow / Keras`
* `VADER & TextBlob`
* `SMOTEENN`
* `Keras Tuner`

🚀 Features
* Detects six labels at once: toxic, severe_toxic, obscene, threat, insult, identity_hate
* Combines TF-IDF with hand-engineered features like sentiment scores and uppercase ratio
* SMOTEENN resampling to fix severe class imbalance before training
* Both models tuned: GridSearchCV for Random Forest, Keras Tuner for the Neural Network
* Confusion matrices and AUC-ROC for honest evaluation

📍 The Process
Most toxic comment projects I came across just dumped text into a basic classifier and moved on.
I wanted to see if adding sentiment signals like VADER scores, punctuation patterns, and text length
would actually help catch what makes a comment feel toxic, not just the words in it.
Built the full pipeline in Colab, dealt with a badly imbalanced dataset using SMOTEENN,
then tuned and compared two models properly. The Neural Network edged out slightly on AUC-ROC,
but the Random Forest was more interpretable. Learned a lot about what "good evaluation" actually means.

🚦 Running the Project
1. Clone the repository
2. Download `train.csv` from the Kaggle Jigsaw Toxic Comment Classification Challenge
3. Upload it to your Colab environment
4. Install dependencies: `pip install vaderSentiment textblob keras-tuner imbalanced-learn`
5. Run all cells sequentially, model and preprocessing files will be saved automatically
