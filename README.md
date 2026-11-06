# Olist Review Score Classification

Predicting customer review scores from Portuguese review text using natural language processing and machine learning.

This project compares **four classification approaches and two regression experiments** for predicting ratings from **1 to 5 stars** using review titles and comments from the Brazilian E-Commerce Public Dataset by Olist.

## Project Objective

Develop a text-based rating prediction pipeline and evaluate both:

- **Exact classification:** accuracy, macro-F1, and per-class precision and recall.
- **Rating distance:** mean absolute error (MAE) and within-one-star accuracy.
- **Continuous regression:** decimal-prediction MAE and root mean squared error (RMSE).

The experiments examine class imbalance, pretrained and learned word representations, ordinal training, and direct numerical prediction. The selected model is the **Embedding CNN classifier**.

## Dataset

**Source:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data)

The project uses `olist_order_reviews_dataset.csv`. Predictive inputs combine `review_comment_title` and `review_comment_message`. Other Olist tables are not required for this text-only task.

The target is `review_score`, from **1 to 5**. Internally, labels are encoded as **0–4**, including the regression targets; regression predictions are shifted back to the star scale for evaluation.

After cleaning, the dataset contains **40,634 reviews** with nonempty comment bodies.

| Subset | Reviews | Approximate proportion |
| --- | ---: | ---: |
| Training | 26,005 | 64% |
| Validation | 6,502 | 16% |
| Test | 8,127 | 20% |

Partitions are stratified by score using `random_state=42`. The vocabulary is adapted using training text only.

## Repository Contents

| File or folder | Purpose |
| --- | --- |
| `Data_Reviews.ipynb` | Explores and cleans reviews, combines text fields, encodes labels, and creates the partitions. |
| `TF_IDF_Logistic_Regression.ipynb` | Implements the TF-IDF baseline with a five-class softmax output, equivalent to multinomial logistic regression. |
| `Fast_Text_CNN.ipynb` | Downloads Portuguese fastText vectors, constructs the embedding matrix, and trains a CNN with frozen embeddings. |
| `Embedding_CNN.ipynb` | Trains and evaluates the learned-embedding CNN classifier and includes the ordinal experiment. |
| `Embedding_CNN_Regression.ipynb` | Tests a learned-embedding CNN with one linear output and MAE loss. |
| `TF_IDF_Ridge_Regression.ipynb` | Tests Ridge regression using sparse TF-IDF features. |
| `Model_Interface.ipynb` | Loads the saved classifier and provides an interface for Portuguese review input. |
| `Olist_Review_Score_Classification.pdf` | Report covering preparation, methods, experiments, results, and limitations. |
| `olist_order_reviews_dataset.csv` | Original review table used for preparation. |
| `splits/train.csv` | Training data for vocabulary adaptation and model fitting. |
| `splits/validation.csv` | Validation data for early stopping and model comparison. |
| `splits/test.csv` | Held-out data for evaluating the selected classifier. |
| `models.zip` | Compressed archive containing only the selected saved Embedding CNN classifier. |
| `README.md` | Project overview and usage instructions. |

The split files contain prepared review text and labels, including `review_comment`, `label`, and `review_score`.

## Models

### 1. TF-IDF + Logistic Regression

Review text → TF-IDF → Dense(5, softmax)

A TF-IDF vector is passed directly to a dense softmax layer without hidden layers. This implements multinomial logistic regression.

### 2. FastText CNN

Review text → Integer tokens → Frozen fastText embedding → Conv1D → Global max pooling → Dropout → Softmax

This model uses pretrained Portuguese fastText vectors with 300 dimensions. Embeddings remain frozen. The implementation loads a fixed word-vector table; it does not dynamically generate subword vectors for unseen words.

### 3. Embedding CNN — Selected Model

Review text → Integer tokens → Trainable embedding → Conv1D → Global max pooling → Dropout → Softmax

Main configuration:

- Vocabulary limit: 15,000 tokens.
- Sequence length: 150 tokens.
- Trainable embedding dimension: 128.
- Convolution: 128 filters, kernel size 3, ReLU activation.
- Global max pooling and dropout of 0.3.
- Five-class softmax output.
- Adam optimizer and sparse categorical cross-entropy.
- Early stopping on validation loss, with best weights restored.
- Square-root balanced class weights.

### 4. Ordinal Embedding CNN

During training, five star probabilities are converted into four cumulative probabilities representing whether the rating exceeds each threshold from one to four stars. Binary cross-entropy compares these probabilities with threshold targets.

At inference, the shared five-class predictor returns star probabilities, and `argmax` selects the most probable star. This is a cumulative ordinal objective, not continuous-output regression.

### 5. Embedding CNN Regression

The learned-embedding CNN retains its feature extraction layers, but replaces the five-class softmax with **one linear output**.

- Loss: **mean absolute error**, not MSE.
- Optimizer: Adam.
- No class weights.
- Early stopping on validation MAE, with best weights restored.

For evaluation, predictions are shifted to the 1–5 scale and clipped to that interval. Decimal predictions are evaluated using MAE and RMSE. Rounded predictions are also evaluated using classification metrics and MAE; half-star ties are rounded upward.

### 6. TF-IDF + Ridge Regression

Review text → TF-IDF → Ridge → Numerical rating

This experiment reuses the training-fitted TF-IDF vectorizer and converts features to sparse matrices in batches. Ridge minimizes squared error with an L2 coefficient penalty, using `alpha=1.0` and `solver="lsqr"`. No alpha search or class weighting is applied. Predictions use the same clipping and rounding procedure as CNN regression.

## Results

### Classification: Validation Comparison

These results use the most probable star, selected with `argmax`.

| Model | Accuracy | Macro-F1 | MAE (stars) |
| --- | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.6906 | ≈0.41 | 0.4882 |
| FastText CNN | 0.6912 | 0.4266 | 0.4843 |
| Embedding CNN | 0.6838 | 0.4511 | 0.4803 |
| Ordinal Embedding CNN | 0.6916 | 0.4306 | 0.4780 |

### Regression: Validation Comparison

All rows refer to the same 6,502 validation reviews. Regression accuracy and macro-F1 use rounded predictions; decimal MAE and RMSE use clipped continuous predictions.

| Model | Accuracy | Macro-F1 | Decimal MAE | RMSE | Integer-prediction MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Embedding CNN classifier (reference) | 0.6838 | 0.4511 | — | — | 0.4803 |
| Embedding CNN regression (MAE loss) | ≈0.56 | ≈0.42 | 0.5963 | 0.8919 | 0.5584 |
| TF-IDF + Ridge regression | ≈0.46 | ≈0.36 | 0.7954 | 1.1609 | 0.7795 |

The tested regression configurations do not improve on the selected classifier. This does not establish that regression generally fails for review ratings: Ridge uses only one alpha value, and these experiments are not an exhaustive model search.

The regression experiments were added after the original classifier test evaluation. Their comparison is **validation-only**; no regression test results are claimed.

### Why Select the Embedding CNN?

The Embedding CNN offers the preferred balance between class recognition and rating-distance error. It has the highest validation macro-F1 among the four classification configurations, while its MAE is only 0.0023 stars above the ordinal model.

Its two-star recall is **0.1298**, compared with **0.0059** for the ordinal model. The ordinal model has slightly better accuracy and MAE but rarely identifies two-star reviews. Both regression configurations have higher integer-prediction MAE and lower macro-F1 than the selected classifier.

Selection therefore considers complementary metrics rather than claiming that one model leads every metric.

### Final Test Evaluation

The selected Embedding CNN is evaluated on **8,127 held-out reviews**.

| Metric | Test result |
| --- | ---: |
| Exact accuracy | 67.82% |
| Macro-F1 | ≈0.45 |
| Weighted F1 | ≈0.65 |
| MAE | 0.4924 stars |
| Within one star | 88.46% |
| Error of exactly one star | 20.64% |
| Error of two or more stars | 11.54% |

F1 values are approximate because the saved classification report displays two decimal places. MAE is rounded to four decimals.

### Confusion Matrix Interpretation

The row-normalized test confusion matrix shows stronger recognition of endpoint ratings:

- **One star:** 85.6% correctly identified.
- **Five stars:** 87.7% correctly identified.
- **Two, three, and four stars:** recall of 13.0%, 23.8%, and 19.2%, respectively.
- 60.3% of two-star reviews are predicted as one star.
- 56.0% of four-star reviews are predicted as five stars.

This explains how modest macro-F1 can coexist with relatively low MAE: many incorrect predictions differ from the observed rating by only one star. However, larger errors remain; 31.7% of three-star reviews are predicted as one star.

The model is useful for approximate rating estimation, but has limited precision in distinguishing intermediate ratings. An MAE near 0.49 does not mean every prediction is within half a star, and the 88.46% within-one-star rate does not imply equally strong performance across classes.

## Saved Model and fastText Files

**Only the selected Embedding CNN classifier is included in `models.zip`.**

The archive is compressed because of file-size constraints. After extraction, the saved file is named:

`Embedidng_CNN.keras`

This spelling matches the saved filename. The other classification and regression models are not included; their experiments can be reproduced using the corresponding notebooks.

**Pretrained fastText vectors are excluded because of their size.** Running the download and embedding-preparation cells in `Fast_Text_CNN.ipynb` retrieves the Portuguese vectors and builds the required embedding matrix.

The selected Embedding CNN does not require external fastText vectors.

## Getting Started

The notebooks are designed for **Google Colab** and use Google Drive paths.

### Reproduce the Experiments

1. Open the notebooks in Colab and mount Google Drive.
2. Place the dataset and split files in Drive and update notebook paths.
3. Run `Data_Reviews.ipynb` to recreate partitions, or use the supplied `splits/` files.
4. Run the classification notebooks: `TF_IDF_Logistic_Regression.ipynb`, `Fast_Text_CNN.ipynb`, and `Embedding_CNN.ipynb`.
5. Run `Embedding_CNN_Regression.ipynb` and `TF_IDF_Ridge_Regression.ipynb` for the additional experiments.
6. Use validation results for development and reserve test data for final evaluation.

The notebooks use their own setup cells; running them in separate sessions requires loading the relevant data and preprocessing objects. Retraining may change results because of initialization, shuffling, dropout, and numerical differences.

### Run the Saved Model

1. Extract `models.zip`.
2. Place `Embedidng_CNN.keras` in your model directory.
3. Open `Model_Interface.ipynb` and update its model path.
4. Load the model and enter a Portuguese title and comment.

```python
from google.colab import drive
import tensorflow as tf
import numpy as np

drive.mount("/content/drive")

model_path = "/content/drive/MyDrive/NLP/models/Embedidng_CNN.keras"
loaded_model = tf.keras.models.load_model(model_path, compile=False)

title = "Muito bom"
comment = "Produto excelente, chegou antes do prazo!"
review = f"{title} {comment}".strip()

probabilities = loaded_model(
    tf.constant([review]), training=False
).numpy()[0]
predicted_score = int(np.argmax(probabilities)) + 1

print("Predicted score:", predicted_score)
print("Probabilities for scores 1–5:", probabilities)
```

The saved model includes its text-vectorization layer and learned parameters. Inference requires neither retraining nor adapting a new vocabulary.

To evaluate the saved model, use the original `splits/test.csv`, preserve row order, and compare predictions with the corresponding labels. Do not shuffle text independently of labels. Loading the exact saved checkpoint avoids the variation introduced by retraining.

## Main Dependencies

- Python
- TensorFlow / Keras
- NumPy
- pandas
- scikit-learn
- SciPy for sparse TF-IDF matrices
- Matplotlib
- ipywidgets for the notebook interface

Google Drive mounting is specific to Colab. Local execution requires adjusting paths and removing Colab-specific cells. Use compatible TensorFlow/Keras versions when loading saved models.

## Limitations

- Ratings are imbalanced; five-star reviews represent approximately half of the cleaned data.
- Intermediate ratings remain difficult to distinguish from text alone.
- Reviews without nonempty comment bodies are excluded.
- CNN inputs are truncated or padded to 150 tokens.
- Splitting is performed at review-row level, without explicitly grouping customers or identical text.
- Results come from individual training runs; small differences do not establish statistical superiority.
- Configurations differ in class weighting, representations, and training objectives, so comparisons do not isolate a single architectural effect.
- Regression results are validation-only, and Ridge's alpha was not tuned.
- Output probabilities have not been established as calibrated confidence estimates.

## Author

**Dongju Mun**  
Instituto Tecnológico y de Estudios Superiores de Monterrey

## References

- [Olist dataset on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data)
- [fastText pretrained word vectors](https://fasttext.cc/docs/en/crawl-vectors.html)
- [Keras TextVectorization documentation](https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/)
- [Keras Embedding documentation](https://keras.io/api/layers/core_layers/embedding/)
- [Scikit-learn Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- Frank, E., & Hall, M. (2001). [A simple approach to ordinal classification](https://ml.cms.waikato.ac.nz/publications/2001/ordinal_tech_report.pdf). University of Waikato. Background for threshold-based ordinal classification; the project's cumulative-softmax implementation is not an exact reproduction of that algorithm.
