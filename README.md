# LLM Guard - A Model to Classify Prompt Injection

According to [OWASP Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/), prompt injection is a type of attack where an adversary crafts input prompts that are intentionally designed to manipulate the behavior of a Large Language Model (LLM), potentially leading to unintended actions or the disclosure of sensitive information.

In this context, the focus of this project is to create a mechanism to detect prompt injection to increase the security of LLM applications. To do so, we tested how different techniques used in Natural Language Representation (NLP) for text representation can be used with classical machine learning algorithms and fine-tuned LLM models to classify user prompts.

# The Dataset

The dataset used in this project was created using the following datasets:

 - [No Robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
 - [The OpenOrca Dataset](https://huggingface.co/datasets/Open-Orca/OpenOrca)
 - [Ultrachat 200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
 - [In-The-Wild Jailbreak Prompts on LLMs](https://github.com/verazuo/jailbreak_llms)
 - [Hackaprompt](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset)

Those datasets contain benign and malicious prompts, mainly in English, and can be found on Huggingface or Github. The resulting dataset has more than 300k prompts, of which 1% is malicious and 99% is benign. This proportion was chosen because we believe that the proportion of malicious prompts is far lesser than the benign. The resulting dataset was split into training, test, and validation and the process used for its creation can be found [here](01%20-%20Data%20Preparation.ipynb).

# Experiments

## TF-IDF

TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a collection (or corpus) of documents. It is given by multiplying the TF and IDF values, which are:

 - **Term Frequency (TF)**: This measures how often a term appears in a document relative to the total number of terms in that document. The assumption is that the more a term appears, the more important it is within that specific document.
 - **Inverse Document Frequency (IDF)**: This measures how important a term is across the entire corpus. It diminishes the weight of common terms and increases the weight of rare terms.
 - **TF-IDF Score**: The final score is calculated by multiplying the TF and IDF values.

**Results:**

| Classifier | Accuracy | Precision | Recall | F1-Score |
| -- | -- | -- | -- | -- |
| Logistic Regression | 0.996044 | 0.942356 | 0.624585 | 0.751249 |
| Random Forest | 0.999730 | 0.977162 | 0.995017 | 0.986008 |
| Gradient Boosting | 0.997363 | 0.937751 | 0.775748 | 0.849091 |
| SVC | 0.998681 | 0.960924 | 0.898671 | 0.928755 |
| MLP | 0.999635 | 0.994872 | 0.966777 | 0.980623 |

The code can be found [here](03%20-%20Prompt%20Injection%20Classification%20with%20TF-IDF.ipynb).

## Word2Vec

Word2Vec is a group of models used to produce word embeddings, which are dense vector representations of words. Developed by Google, it captures semantic meanings and relationships between words based on their context in large text corpora.

There are two main architectures for Word2Vec:

 - **Continuous Bag of Words (CBOW)**: This model predicts a target word based on its surrounding context words. It takes context words as input and tries to predict the center word.
 - **Skip-Gram**: This model works in the opposite direction; it uses a target word to predict the surrounding context words. It is especially effective for inferring relationships between less frequent words.

**Results:**

| Classifier | Accuracy | Precision | Recall | F1-Score |
| -- | -- | -- | -- | -- |
LogisticRegression | 0.995488 | 0.856502 | 0.634551 | 0.729008 |
RandomForest | 0.999825 | 0.985222 | 0.996678 | 0.990917 |
GradientBoosting | 0.996092 | 0.908257 | 0.657807 | 0.763006 |
SVC | 0.997283 | 0.937120 | 0.767442 | 0.843836 |
MLP | 0.999396 | 0.971572 | 0.965116 | 0.968333 |

The code can be found [here](04%20-%20Prompt%20Injection%20Classification%20with%20Word2Vec.ipynb).

## Bert

BERT is a transformer-based model developed by Google that is designed to understand the context of words in a sentence by considering the entire sentence rather than a fixed window of context. It is trained on a large corpus of text and employs a two-step process:

 - **Pre-training:** BERT is trained on a masked language modeling task, where some words in the input are masked, and the model learns to predict them based on their context. It also uses next sentence prediction to understand relationships between sentences.
 - **Fine-tuning:** After pre-training, BERT can be fine-tuned on specific tasks (e.g., question answering, sentiment analysis) with task-specific data.

**Results:**

| Classifier | Accuracy | Precision | Recall | F1-Score |
| -- | -- | -- | -- | -- |
| Fine-tunned Bert | 0.998252 | 0.964150 | 0.848837 | 0.902826 |

The code can be found [here](05%20-%20Prompt%20Injection%20Classification%20with%20BERT.ipynb).

# Conclusion

The accuracy values achieved by the three different techniques are quite similar. Surprisingly, the more advanced methods that capture the semantic meanings and relationships between words in their context did not enhance the overall performance of the trained model. In fact, the pre-trained LLM model BERT demonstrated a lower recall rate compared to the simpler TF-IDF technique. This observation suggests that classifying malicious prompts may be relatively simple, as it can often be reduced to counting the frequency of specific words or characters in a sentence. However, it is crucial to acknowledge that, although the classification process may appear straightforward and the TF-IDF method can effectively classify prompts, the complexities of language and the potential for more sophisticated prompt injection techniques can complicate this task.