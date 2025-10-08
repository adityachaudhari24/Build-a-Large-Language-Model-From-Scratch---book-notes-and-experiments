# Build-a-Large-Language-Model-From-Scratch---book-notes-and-experiments
The models you’ll create are smaller in scale compared to the large foundational models; they use the same concepts and serve as powerful educational tools to grasp the core mechanisms and techniques used in building state-of-the-art LLMs.

#### Majority of the content in this repo is from the book "Build a Large Language Model from Scratch" by Sebastian Raschka. All credits to the author for the amazing work. Some text is directly copied from the book and some is tuned for understanding concept better.

#### executing below steps in the book
![alt text](image-6.png)


# Chapter 1: understanding the large language models
<details>
<summary>🎯Q. LLMs vs Earlier NLP</summary>

- **LLMs (Large Language Models)** are transformer-based deep learning models trained on massive text datasets. They excel at generating context-aware, coherent text but lack human-level consciousness.
- **Earlier NLP Models** relied on:
  - Handcrafted rules or simpler algorithms for narrow tasks (e.g., spam detection, translation).
  - Focused on pattern recognition and categorization but struggled with nuanced, complex tasks.
- **Key Difference**: LLMs, powered by `transformers` and `large datasets`, represent a paradigm shift. They are versatile and capable of handling complex language understanding and generation tasks.

⭐ **Takeaway**: Transformers and massive datasets have revolutionized NLP, making it more flexible and powerful. ⭐
</details>


<details>
<summary>🎯Q. What is an LLM </summary>

- An LLM is a ⭐neural network⭐ designed to understand, generate, and respond to human-like text.
- Models like this often have tens or even hundreds of billions of parameters, which are the adjustable weights in the network that are optimized during training to predict the next word in a sequence.
- The “large” in “large language model” refers to both the model’s size in terms of parameters and the immense dataset on which it’s trained.
- Think of `parameters` as the number of knobs you can turn to adjust the model's behavior, and `training data` is like the vast library of books and articles the model reads to learn language patterns. 

- LLMs utilize an architecture called the `transformer`, which allows them to pay selective attention to different parts of the input when making predictions, making them
especially adept at handling the nuances and complexities of human language.

- ⭐⭐Machine learning⭐ and ⭐deep learning⭐ are fields aimed at implementing algorithms that enable computers
to learn from data and perform tasks that typically require human intelligence.⭐

- traditional machine learning requires a manual feature engineering step, where domain experts identify and extract relevant features from the data to train models. In contrast, deep learning models automatically learn hierarchical feature representations from raw data through multiple deep layers, eliminating the need for manual feature engineering.

![alt text](image.png)
</details>

<details>
<summary>🎯Q. Stages of building and using LLMs</summary>

- The process of creating an LLM involves two main stages: `pretraining` and `fine-tuning`.

### 🎯 Traditional Machine Learning vs. Pretraining and Fine-Tuning in LLMs:
1. **Traditional Machine Learning**:
   - Uses `supervised learning`, where models are trained from scratch for a specific task using labeled data.
2. **LLM Pretraining**:
   - Relies on `self-supervised learning`, where the model learns general language patterns from massive amounts of unlabeled text data.
3. **LLM Fine-Tuning**:
   - Applies `supervised learning` on smaller, labeled datasets to adapt the model for specific tasks.

### Types of Fine-Tuning:
- **Instruction Fine-Tuning**:
  - Uses datasets with input-output pairs (e.g., questions and answers) to teach the model specific tasks.
- **Classification Fine-Tuning**:
  - Uses labeled datasets with text and class labels (e.g., emails labeled as “spam” or “not spam”).
  - This approach is common for tasks like sentiment analysis or topic classification.

</details>


<details>
<summary>🎯Q. Overview of the transformer architecture - key notes</summary>

- Most modern LLMs rely on the transformer architecture, which is a 💡`deep neural network architecture`💡 introduced in the 2017 paper “Attention Is All You Need”
- To understand LLMs, we must understand the original transformer, which was developed for machine translation, translating English texts to German and French.
- There are key two steps in the transformer architecture:
  1. The `encoder` processes the input text and creates a numerical representation of it (Embeddings) ⭐which capture the contextual information of the input.⭐
  2. The `decoder` takes this representation initially convert literally word by word  and then goes through `self-attention` and then generates the final output text.
- This image is simple initial depiction however many things goes inside which autor is about to explain in the next chapters.
![alt text](image-1.png)

- After this paper on this concept other variants transformer emerged and become backbone of many LLMs. Like BERT (short for bidirectional
encoder representations from transformers) and the various GPT models (short for generative pretrained transformers)
</details>

<details>
<summary>🎯Q. A closer look at the GPT architecture</summary>

- GPT was originally introduced in the paper “Improving Language Understanding by Generative Pre-Training” in 2018 by Alec Radford and colleagues at OpenAI.
- The model is simply trained to preduct the next --> word in a sequence of words.
- Compared to the original transformer architecture we covered in section 1.4, the general GPT architecture is relatively simple. Essentially, it’s just the decoder part without the encoder.
- Since decoder-style models like GPT generate text by predicting text one word at a time, they are considered a type of `autoregressive model`.
- `Autoregressive models` incorporate their previous outputs as inputs for future predictions.
- The ability to perform tasks that the model wasn’t explicitly trained to perform is called an `emergent behavior`. This capability isn’t explicitly taught during training but emerges as a natural consequence of the model’s exposure to vast quantities of multilingual data in diverse contexts.

</details>


<details>
<summary>🎯Q. Building a large language model - stages (section 1.7)</summary>

- from the book - "Build a Large Language Model from Scratch" by Sebastian Raschka
- ![alt text](image-2.png)
</details>



# Chapter 2: working with text data

<details>
<summary>🎯Q. understanding embeddings</summary>

- The concept of converting data into a vector format is often referred to as embedding.
- It’s important to note that different data formats require distinct embedding models. For example, an embedding model designed for text would not be suitable for embedding audio or video data.
- ⭐The primary purpose of embeddings is to convert nonnumeric data into a format that neural networks can process.⭐

</details>

<details>
<summary>🎯Q. Key steps in data preparation which includes embeddings - IMPORTANT</summary>

- All the steps along with test code is mentiond under @chp_02_embeddings/embeddings.ipynb file. Please have a look.
- Below is the summary of the steps.
  1. `Tokenization` - breaking down text into smaller units called tokens.
      - Example:
          - Input: "The cat sat on the mat."
          - Output Tokens: ["The", "cat", "sat", "on", "the", "mat", "."]
  2. `Token IDs` - converting tokens into numerical representations called token IDs.
      - Example:
          - Tokens: ["The", "cat", "sat", "on", "the", "mat", "."]
          - Token IDs: [101, 2009, 2021, 2006, 1996, 3007, 1012]
  3. `Token Embeddings` - transforming token IDs `into dense vectors` using an embedding layer.
      - Example:
          - Token IDs: [101, 2009, 2021, 2006, 1996, 3007, 1012]
          - Token Embeddings: [[0.1, 0.2, ...], [0.3, 0.4, ...], ..., [0.5, 0.6, ...]]
  4. `Positional Embeddings` - adding positional information to token embeddings to capture word order.
  5. `Input Embeddings` - combining `token embeddings` and `positional embeddings` to create input embeddings for the model.
  6. Finally this input embeddings are fed into the transformer model for further processing. (Chp 3 onwards)

</details>

<br>
<br>
<br>

# Chapter 3: coding attention mechanisms.

<details>
<summary>🎯Q. why we need the attention mechanism in LLM ?</summary>

- The attention mechanism allows the model to focus on different parts of the input sequence when making predictions, rather than treating all parts equally.
- This is especially important for handling long-range dependencies in text, where the meaning of a word can depend on words that are far apart in the sequence.
- The attention mechanism helps the model to dynamically weigh the importance of different words in the input sequence based on their relevance to the current prediction task.
- This allows the model to capture context more effectively and generate more coherent and contextually relevant text.
- Example : Suppose we want to develop a language translation model that translates text from one language into another. As shown in figure below, we can’t simply translate a text word by word due to the grammatical structures in the source and target language.
- ![alt text](image-7.png)
- RNN (recurrent neural network) is usually used for this kind of tasks but it has its own limitations like vanishing gradient problem, long training times etc. To overcome these limitations attention mechanism is used.
    - ![alt text](image-8.png)
- ⭐ The big limitation of encoder–decoder RNNs is that the RNN can’t directly access earlier hidden states from the encoder during the decoding phase. Consequently, it relies solely on the current hidden state, which encapsulates all relevant information. This can lead to a loss of context, especially in complex sentences where dependencies might span long distances.⭐ 
- Fortunately, it is not essential to understand RNNs to build an LLM. Just remember that encoder–decoder RNNs had a shortcoming that motivated the design of
attention mechanisms.
- Later, researchers discovered that RNNs were not essential for NLP deep neural networks and introduced the transformer architecture with self-attention, inspired by the Bahdanau attention mechanism.
</details>


<details>
<summary>🎯Q. What is self attention mechanism actually ?</summary>

- Self-attention is a mechanism that allows each position in the input sequence to consider the relevancy of, or “attend to,” all other positions in the same sequence when computing the representation of a sequence.
- Self-attention is a key component of contemporary LLMs based on the transformer architecture, such as the GPT series.
- The `self-attention mechanism` enables the model to weigh the importance of different words in a sequence relative to each other, allowing it to capture context and relationships more effectively.
- In `self-attention`, the `“self”` refers to the mechanism’s ability to compute attention weights by relating different positions within a single input sequence.
- It assesses and learns the relationships and dependencies between various parts of the input itself, such as words in a sentence or pixels in an image.
- This is in contrast to traditional attention mechanisms, where the focus is on the relationships between elements of two different sequences.
- ⭐⭐In the context of self-attention mechanisms, the dot product determines the extent to which each element in a sequence focuses on, or “attends to,” any other element: the higher the dot product, the higher the similarity and attention score between two elements.⭐⭐
</details>

<details>
<summary>🎯Q. 3 step process for self-attention ?</summary>

- Its a 3 step process to calculate self attention
  1. step 1 - compute unnormalized attention scores "w"
  2. step 2 - normalize the unnormalized attention scores ("omegas","w") so that they sum up to 1
  3. step 3 - compute the context vector by multiplying the embedded input tokens, with the attention weights and sum the resulting vectors.
- All the steps along with test code is mentioned under @chp_03_self_attention_theory/self_attention.ipynb file. Please have a look. This is super interesting and the original author has explained it very well.
- ![alt text](image-9.png)

</details>

<br>
<br>
<br>


# Chapter 4: Implementing a GPT model from scratch to generate text

<details>
<summary>🎯Q.what is the meaning of "parameter" in GPT like LLM ?</summary>

- In the context of deep learning and LLMs like GPT, the term “parameters” refers to the trainable weights of the model. (weights is also a n-d array/tensor)
- These weights are essentially the internal variables of the model that are adjusted and optimized during the training process to `minimize a specific loss function`. Meaning the model learns to make better predictions by adjusting these weights based on the input data and the expected output.
- ![alt text](image-14.png)
- ![alt text](image-15.png)

- In a neural network, parameters include:
  - `Weights`: These are the coefficients(a numerical or constant) that connect neurons in one layer to neurons in the next layer. They determine the strength and direction of the connection.
  - `Biases`: These are additional parameters added to the weighted sum of inputs to a neuron before applying the activation function. They help the model learn patterns that do not pass through the origin. (activatoin function is like a threshold function)
- The number of parameters in a model is often used as a measure of its complexity and capacity. Larger models with more parameters can potentially learn more complex patterns from data, but they also require more computational resources and data to train effectively.
- In the context of LLMs like GPT, the number of parameters can range from millions to billions, depending on the size and architecture of the model.
</details>

<details>
<summary>🎯Q.what is the transformer block ?</summary>

- Transformer block actually is a combination of multiple layers and components that work together to process and understand sequential data, such as text.
- A transformer block is a fundamental `building unit of the transformer architecture`, which is widely used in natural language processing (NLP) tasks, including large language models (LLMs) like GPT.
- A transformer block typically consists of two main components:
  1. `Multi-Head Self-Attention Mechanism`: This component allows the model to focus on different parts of the input sequence when making predictions. It computes attention scores for each token in the sequence relative to all other tokens, enabling the model to capture context and relationships effectively.
  2. `Feed-Forward Neural Network`: This component processes the output of the self-attention mechanism, applying a series of transformations to enhance the model's ability to learn complex patterns.
- ![alt text](image-16.png)
</details>


<details>
<summary>🎯Q. Summary of the chap 4 ?</summary>

- summary ![alt text](image-17.png)
</details>



<br>
<br>
<br>

# Chapter 5: pretraining on unlabeled text data.

- chapter covers, basic model evaluation techniques, loading pretrained weights for good start for our model for fine tuning, and generating text with pretrained models.
- ![alt text](image-18.png)

<details>
<summary>🎯Q. what is weight parameter ?</summary>

- ![alt text](image-19.png)
</details>



<br><br>
<br>
<br>
⭐⭐⭐ GENERAL NOTES ⭐⭐⭐

- Research has shown that when it comes to modeling performance, custom-built
LLMs—those tailored for specific tasks or domains—can outperform general-purpose
LLMs, such as those provided by ChatGPT or GPT-4. This is because custom models can be fine-tuned to better understand the nuances and specific requirements of a given task, leading to improved accuracy and effectiveness.

<details>
<summary>🎯Q. Explain back-propogation in super simple terms ?</summary>

- The backpropagation algorithm is a method used to train neural networks. It helps the network learn from its mistakes and improve its predictions over time. Here’s a simple breakdown of how it works:

1. `Prediction`: The neural network makes a guess (output) based on the input data.
2. `Error Calculation`: The network checks how wrong its guess was by comparing it to the correct answer (using a loss function).
3. `Blame Assignment`: Backpropagation works backward through the network, figuring out which parts (neurons/weights) contributed most to the error.
4. `Adjustment`: The network adjusts its weights (parameters) slightly to reduce the error for the next prediction. This is done using a method called `gradient descent`.
5. `Repeat`: This process repeats many times, improving the network's predictions step by step.
</details>

<details>
<summary>🎯Q. Difference between LLM vs foundation model ?</summary>

- LLM is a type of foundation model specifically designed for natural language processing tasks. Foundation models are large-scale models trained on vast amounts of data that can be adapted to various tasks, including but not limited to language processing.
- Foundation model servers as a base for the fine-tuning and adaptation to specific tasks, while LLMs are specialized versions of these models focused on understanding and generating human language.
</details>

<details>
<summary>🎯Q. Does attention mechanism and positional encoding complement each other ? how?</summary>
- ![alt text](image-10.png)
</details>

<details>
<summary>🎯Q. How self-attention is different then positional encoding ?</summary>

- The difference between self-attention and positional encoding lies in their roles:

    - `Self-Attention`: Focuses on the relationships between tokens in a sequence, determining which tokens are most relevant to each other, regardless of their positions. It captures context dynamically.

    - `Positional Encoding`: Provides information about the order of tokens in the sequence, ensuring the model understands the sequence structure since transformers process tokens in parallel.

- In short, self-attention captures "what to focus on," while positional encoding provides "where in the sequence."
- ![alt text](image-11.png)
- ![alt text](image-12.png)

</details>


<details>
<summary>🎯Q. how does training looks before foundation model and after foundation model ? good high level diagram </summary>

- ![alt text](image-20.png)
</details>

<br>
<br>
<br>

# Maths concepts for AI,ML and DL

</details>


<details>
<summary>🎯Q. what is tensor?</summary>

- Tensors represent a mathematical concept that generalizes vectors and matrices to potentially higher dimensions.
- In other words, tensors are mathematical objects that can be characterized by their order (or rank), which provides the number of dimensions. For example, a scalar (just a number) is a tensor of rank 0, a vector is a tensor of rank 1, and a matrix is a tensor of rank 2
- 🔥From a computational perspective, tensors serve as `data containers`🔥. For instance, they hold multidimensional data, where 🔥each dimension represents a different feature🔥.
- Tensor libraries like `PyTorch` can create, manipulate, and compute with these arrays efficiently. In this context, `a tensor library functions as an array library`.
- PyTorch tensors are similar to NumPy arrays but have several additional features that are important for deep learning.
- ![alt text](image-4.png)
- ![alt text](image-3.png)
- ![alt text](image-5.png)

- ref : https://www.kdnuggets.com/2018/05/wtf-tensor.html
</details>

<br>
<br>
<br>

#### Questions which needs a deep dive

<details>
<summary>🎯Q. If parameters in any model store the knnowledge can we see each parameter on which knowledge they have stored?</summary>

- This is integpretability challenge, one of the biggest open problems in AI research (Rudin, 2019; Castelvecchi, 2016)
- TODO - do understand and go deep.
</details>

<details>
<summary>🎯Q. Why positional encoding is needed even after token IDs encodings into vectors? chp 2 section 2.8 </summary>

- TODO
</details>



<br>
<br>
<br>
***********
<details>
<summary>Emojis used</summary>
⭐ - For important points
🔥 - super important
💡 - For key concepts/tips
⚠️ - For warnings/common mistake
🎯 - For exam targets/focus areas/ question 
🚀 - For advanced topics .
🚫 - For indicating something that cannot be used or a concerning point
</summary>
</details>