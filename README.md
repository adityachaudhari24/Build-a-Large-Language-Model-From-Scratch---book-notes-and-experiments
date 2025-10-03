# Build-a-Large-Language-Model-From-Scratch---book-notes-and-experiments
The models you’ll create are smaller in scale compared to the large foundational models; they use the same concepts and serve as powerful educational tools to grasp the core mechanisms and techniques used in building state-of-the-art LLMs.

#### Majority of the content in this repo is from the book "Build a Large Language Model from Scratch" by Sebastian Raschka. All credits to the author for the amazing work. Some text is directly copied from the book for and some is tuned for my better understanding.

# Chapter 1: understanding the large language models
<details>
<summary>🎯Q. LLMs vs earlier NLP </summary>

- An LLM is a specific type of model within the broader field of NLP.

- LLMs are transformer-based deep models trained on massive text corpora that deliver broad, context-aware language abilities far beyond earlier task-specific or rule-based NLP systems — though “understand” here means producing coherent, contextually relevant text, not human-level consciousness.

- Large language models (LLMs), built on deep neural networks and powered by the transformer architecture, represent a major shift in natural language processing (NLP). Unlike earlier NLP approaches, which relied on handcrafted rules or simpler models tailored to narrow tasks like spam detection or translation, LLMs are trained on vast amounts of text data.

- While earlier NLP models excelled at categorization and pattern recognition, they struggled with complex understanding and generation. LLMs, by contrast, demonstrate remarkable versatility, though their “understanding” refers to producing text that appears meaningful and relevant.
⭐This paradigm shift—driven by transformers and massive datasets—has fundamentally transformed NLP into a far more capable and flexible field.⭐
</details>


<details>
<summary>🎯Q. What is an LLM </summary>

- An LLM is a neural network designed to understand, generate, and respond to human-like text.
- Models like this often have
tens or even hundreds of billions of parameters, which are the adjustable weights in
the network that are optimized during training to predict the next word in a sequence.
- The “large” in “large language model” refers to both the model’s size in terms of parameters and the immense dataset on which it’s trained.
- Think of `parameters` as the number of knobs you can turn to adjust the model's behavior, and training data is like the vast library of books and articles the model reads to learn language patterns. 

- LLMs utilize an architecture called the `transformer`, which allows them to pay selective attention to different parts of the input when making predictions, making them
especially adept at handling the nuances and complexities of human language.

- ⭐⭐Machine learning⭐ and ⭐deep learning⭐ are fields aimed at implementing algorithms that enable computers
to learn from data and perform tasks that typically require human intelligence.⭐

- traditional machine learning requires a manual feature engineering step, where domain experts identify and extract relevant features from the data to train models. In contrast, deep learning models automatically learn hierarchical feature representations from raw data through multiple deep layers, eliminating the need for manual feature engineering.

![alt text](image.png)
</details>

<details>
<summary>🎯Q. Stages of buiding and using the LLMs </summary>

- The general process of creating an LLM includes `pretraining` and `fine-tuning`.
- 🎯traditional machine learning vs pretraining and fine-tuning of LLMs🎯:
  - In traditional `machine learning`, it uses `supervised learning`, where models are trained from scratch on a single, specific task using labeled data.
  - In `pretraning` an LLM large `self-supervised` learning is used, where the model learns general language patterns from massive amounts of unlabeled text data. 
  - In `fine-tuning` the LLMs it uses the `supervised learning on smaller, labeled datasets to adapt the model for specific tasks`.

  - The two most popular categories of fine-tuning LLMs are `instruction fine-tuning` and `classification fine-tuning`.
    - In `instruction fine-tuning`, the labeled dataset consists of instruction and answer pairs, such as a query to translate a text accompanied by the correctly translated text. 
    - In `classification fine-tuning`, the labeled dataset consists of texts and associated class labels—for example, emails associated with “spam” an“not spam” labels.
</details>


<details>
<summary>🎯Q. Transformer architecture - key notes</summary>

- Most modern LLMs rely on the transformer architecture, which is a deep neural network architecture introduced in the 2017 paper “Attention Is All You Need”
- To understand LLMs, we must understand the original transformer, which was developed for machine translation, translating English texts to German and French.
- There are key two steps in the transformer architecture:
  1. The `encoder` processes the input text and creates a numerical representation of it (Embeddings) which capture the contextual information of the input.
  2. The `decoder` takes this representation initially convert literally word by word  and then goes through self-attention and then generates the final output text.
- This image is simple initial depiction however many things goes inside which autor is about to explain in the next chapters.
![alt text](image-1.png)

- After this paper on this concept other variants transformer emerged and become backbone of many LLMs. Like BERT (short for bidirectional
encoder representations from transformers) and the various GPT models (short for genera-
tive pretrained transformers)
</details>

<br>
<br>
<br><br>
<br>
<br><br>
<br>
<br>
⭐⭐⭐ GENERAL NOTES ⭐⭐⭐
- Research has shown that when it comes to modeling performance, custom-built
LLMs—those tailored for specific tasks or domains—can outperform general-purpose
LLMs, such as those provided by ChatGPT or GPT-4. This is because custom models can be fine-tuned to better understand the nuances and specific requirements of a given task, leading to improved accuracy and effectiveness.

-

<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
***********
<details>
<summary>Emojis used</summary>
⭐ - For important points
🔥 - For hot/important exam topics
💡 - For key concepts/tips
⚠️ - For warnings/common mistake
🎯 - For exam targets/focus areas/ question 
🚀 - For advanced topics .
🚫 - For indicating something that cannot be used or a concerning point
</summary>
</details>