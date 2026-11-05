# Olist Review Score Classification

Predicting customer review scores from Portuguese review text using natural language processing and machine learning.

This project compares four approaches to predicting ratings from **1 to 5 stars** using the review title and comment from the Brazilian E-Commerce Public Dataset by Olist.

## Project Objective

Develop a text classification pipeline that estimates the rating associated with a customer review and evaluates both:

- **Exact classification**, using accuracy and macro-F1.
- **Rating distance**, using mean absolute error (MAE).

The experiments examine class imbalance, pretrained and learned word representations, and ordinal training.

## Dataset

**Source:** [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data)

The project uses `olist_order_reviews_dataset.csv`.

The predictive input combines:

- `review_comment_title`
- `review_comment_message`

The target is `review_score`, which ranges from **1 to 5**. Internally, labels are encoded as **0–4** for training.

After cleaning, the dataset contains **40,634 reviews** with nonempty comment bodies.

| Subset | Reviews | Approximate proportion |
| --- | ---: | ---: |
| Training | 26,005 | 64% |
| Validation | 6,502 | 16% |
| Test | 8,127 | 20% |

The partitions are stratified by review score. The vocabulary is adapted using training text only.

## Repository Contents

| File or folder | Purpose |
| --- | --- |
| `Data_Reviews.ipynb` | Loads and explores the review data, handles missing and duplicated observations, combines the text fields, encodes labels, and creates the training, validation, and test partitions. |
| `TF_IDF_Logistic_Regression.ipynb` | Implements the TF-IDF baseline with a five-class softmax output, equivalent to multinomial logistic regression. |
| `Fast_Text_CNN.ipynb` | Downloads Portuguese fastText vectors, builds an embedding matrix for the training vocabulary, and trains a CNN with frozen pretrained embeddings. |
| `Embedding_CNN.ipynb` | Trains the CNN with learned embeddings, evaluates it, and includes the ordinal CNN experiment. |
| `Model_Interface.ipynb` | Loads the saved Embedding CNN and provides an interface for entering Portuguese review text and obtaining a predicted rating. |
| `Olist_Review_Score_Classification.pdf` | Project report covering data preparation, methodology, experiments, results, and limitations. |
| `olist_order_reviews_dataset.csv` | Original review table used for data preparation. |
| `splits/` | Prepared CSV partitions used by the modeling notebooks. |
| `splits/train.csv` | Training data used to learn the vocabulary and model parameters. |
| `splits/validation.csv` | Validation data used for early stopping and model comparison. |
| `splits/test.csv` | Held-out data used for the final evaluation of the selected model. |
| `models.zip` | Compressed archive containing the saved Embedding CNN. |
| `README.md` | Project overview and usage instructions. |

The split files contain the prepared review text and labels, including `review_comment`, `label`, and `review_score`.

## Models

### 1. TF-IDF + Logistic Regression

The review text is converted into a TF-IDF vector and passed directly to a dense softmax layer with five outputs.

**Pipeline**

Review text → TF-IDF → Dense(5, softmax)

### 2. Fast_Text_CNN

This model uses pretrained Portuguese fastText word vectors with 300 dimensions. The embedding layer remains frozen during training.

**Pipeline**

Review text → Integer tokens → Frozen fastText embedding → Conv1D → Global max pooling → Dropout → Softmax

### 3. Embedding CNN

This model learns 128-dimensional word embeddings directly from the review-score prediction task.

**Pipeline**

Review text → Integer tokens → Trainable embedding → Conv1D → Global max pooling → Dropout → Softmax

Main configuration:

- Vocabulary limit of 15,000 tokens.
- Sequence length of 150 tokens.
- Embedding dimension of 128.
- 128 convolution filters with kernel size 3 and ReLU activation.
- Global max pooling.
- Dropout rate of 0.3.
- Five-class softmax output.
- Adam optimizer.
- Sparse categorical cross-entropy.
- Early stopping based on validation loss.
- Square-root balanced class weights.

### 4. Ordinal Embedding CNN

This experiment uses the ordered nature of the ratings.

During training, the five star probabilities are converted into four cumulative probabilities representing whether the rating exceeds each threshold from one to four stars. These outputs are trained using binary cross-entropy.

At inference, the model returns five star probabilities, and the most probable star is selected.

## Results

### Validation Comparison

The following results use the most probable star, selected with `argmax`.

| Model | Accuracy | Macro-F1 | MAE |
| --- | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.6906 | ≈0.41 | 0.4882 |
| Fast_Text_CNN | 0.6912 | 0.4266 | 0.4843 |
| Embedding CNN | 0.6838 | 0.4511 | 0.4803 |
| Ordinal Embedding CNN | 0.6916 | 0.4306 | 0.4780 |

### Selected Model

The **Embedding CNN** is selected because it provides the preferred balance between class recognition and rating-distance error.

It achieves the highest validation macro-F1 among the four configurations while maintaining an MAE close to the ordinal model. Its two-star recall is **0.1298**, compared with **0.0059** for the ordinal model.

The ordinal model obtains slightly better accuracy and MAE, but rarely identifies two-star reviews correctly.

### Final Test Evaluation

The selected Embedding CNN is evaluated on **8,127 held-out reviews**.

| Metric | Test result |
| --- | ---: |
| Accuracy | ≈0.68 |
| Macro-F1 | ≈0.45 |
| Weighted F1 | ≈0.65 |
| MAE | 0.4924 stars |

Accuracy and F1 values are approximate because the saved classification report displays two decimal places.

An MAE of approximately **0.49 stars** indicates useful average rating proximity. However, the model performs substantially better on one- and five-star reviews than on intermediate ratings. It should therefore be interpreted as an approximate rating estimator with limited fine-grained class recognition.

## Saved Model and fastText Files

**Only the selected Embedding CNN is included in `models.zip`.**

The archive is compressed to accommodate file-size constraints. After extraction, the saved model file is:

`Embedidng_CNN.keras`

The saved TF-IDF, fastText CNN, and ordinal CNN models are not included. Their experiments can be reproduced through the corresponding notebooks.

**The pretrained fastText vectors are also excluded because of their size.** Running the download and embedding-preparation cells in `Fast_Text_CNN.ipynb` retrieves the Portuguese vectors and builds the required embedding matrix.

The selected Embedding CNN does not require the external fastText vectors.

## Getting Started

The notebooks are designed for **Google Colab** and use Google Drive paths.

### Reproduce the Experiments

1. Open the notebooks in Google Colab.
2. Mount Google Drive when prompted.
3. Place the dataset and split files in your Drive.
4. Update the file paths to match your folder structure.
5. Run `Data_Reviews.ipynb` if you want to recreate the prepared partitions.
6. Run the modeling notebooks in the following order:
   - `TF_IDF_Logistic_Regression.ipynb`
   - `Fast_Text_CNN.ipynb`
   - `Embedding_CNN.ipynb`
7. Use validation results for model development and reserve the test set for final evaluation.

The prepared files in `splits/` allow you to run the modeling notebooks without repeating data preparation.

### Run the Saved Model

1. Extract `models.zip`.
2. Place `Embedidng_CNN.keras` in your model directory.
3. Open `Model_Interface.ipynb`.
4. Update its loading path to match the extracted model.
5. Run the notebook and enter a Portuguese review.

Example loading code:

```python
import tensorflow as tf

model_path = "/content/drive/MyDrive/NLP/models/Embedidng_CNN.keras"

loaded_model = tf.keras.models.load_model(
    model_path,
    compile=False
)
```

Example prediction:

```python
import numpy as np
import tensorflow as tf

review = "Produto excelente, chegou antes do prazo!"

probabilities = loaded_model(
    tf.constant([review]),
    training=False
).numpy()[0]

predicted_score = int(np.argmax(probabilities)) + 1

print("Predicted score:", predicted_score)
print("Probabilities for scores 1–5:", probabilities)
```

The saved model includes its text-vectorization layer and learned parameters, so inference does not require retraining or adapting a new vocabulary.

## Main Dependencies

- Python
- TensorFlow / Keras
- NumPy
- pandas
- scikit-learn
- Matplotlib
- ipywidgets for the notebook interface

Google Drive mounting is specific to Colab. Local execution requires adjusting the paths and removing Colab-specific cells.

## Limitations

- The rating distribution is imbalanced, with five-star reviews representing approximately half of the cleaned dataset.
- Intermediate ratings remain difficult to distinguish from text alone.
- Reviews without nonempty comment bodies are excluded.
- CNN inputs are truncated or padded to 150 tokens.
- The partition is performed at the review-row level rather than explicitly grouping customers or identical text.
- Results come from individual training runs; small differences do not establish statistical superiority.
- Output probabilities have not been established as calibrated confidence estimates.

## Author

**Dongju Mun**  
Instituto Tecnológico y de Estudios Superiores de Monterrey

## References

- [Olist dataset on Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data)
- [fastText pretrained word vectors](https://fasttext.cc/docs/en/crawl-vectors.html)
- [Keras TextVectorization documentation](https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/)
