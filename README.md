The automatic generation of natural language descriptions for images is a challenging and important task at the intersection of computer vision (CV) and natural language processing (NLP). This task, known as image captioning, aims to generate concise, human-like textual descriptions that accurately reflect the objects, actions, and relationships depicted in a visual scene. Such technology has numerous practical applications, including assisting visually impaired individuals through auditory scene descriptions, improving content-based image retrieval systems, enabling more effective human-robot interaction, and enhancing the accessibility of visual media on the web.

Traditional image understanding approaches primarily focused on object detection and scene classification. However, image captioning requires a deeper level of understanding, moving beyond simple labels to generate syntactically correct and semantically meaningful sentences. The emergence of deep learning has significantly advanced this field, with encoder-decoder architectures becoming the dominant approach.

The task of automatically generating image captions has evolved rapidly alongside advancements in deep learning. One of the most influential works in this area is "Show and Tell" by Vinyals et al. [1], which introduced a CNN-LSTM encoder-decoder architecture. The model used a convolutional neural network to encode an image into a fixed-length feature vector and an LSTM network to generate a caption word by word. This approach demonstrated the ability to produce more natural and contextually relevant captions compared to previous methods.

Subsequent research introduced attention mechanisms, such as those proposed in "Show, Attend and Tell" by Xu et al. [2]. Instead of relying on a single image representation, attention mechanisms allow the decoder to focus on different image regions while generating each word, resulting in more detailed and accurate captions. More recently, Transformer-based models [3] have achieved state-of-the-art performance by relying entirely on attention mechanisms and capturing long-range dependencies more effectively.

Further demonstrating the effectiveness of CNN-LSTM architectures, Meghana and Sekharbabu [4] explored an image caption generation framework utilizing VGG16 for feature extraction and LSTM for caption generation on the Flickr8k dataset. Their findings confirm that the VGG16-LSTM combination remains a strong baseline for image captioning tasks.

Our work follows the established CNN-LSTM encoder-decoder paradigm and provides a solid baseline implementation. Although attention-based and Transformer-based approaches offer opportunities for further improvement, the chosen architecture remains a reliable and effective solution for image caption generation.

The Flickr8k dataset was used for training and evaluating the proposed image captioning model. The dataset contains 8,091 images, each associated with five human-generated captions. A standard data split was employed: 70% of images (5,663) for training, 15% (1,212) for validation, and 15% (1,216) for testing.

Visual features were extracted using the VGG16 model pre-trained on the ImageNet dataset. The final classification layer was removed, and the output of the penultimate fully connected layer (fc2) was used as a 4096-dimensional feature vector representing each image. Prior to feature extraction, all images were resized to 224×224 pixels and preprocessed according to VGG16 requirements.

Text preprocessing involved converting captions to lowercase, removing non-alphabetic characters, tokenizing words, adding special startseq and endseq tokens, padding sequences to a fixed maximum length, and converting target words into one-hot encoded vectors. The vocabulary size was determined based on the unique words appearing in the training captions.

The proposed image caption generator follows an encoder-decoder framework. The encoder consists of the pre-trained VGG16 network, which transforms an image into a fixed-length 4096-dimensional feature vector representing its visual content.

The decoder is based on an LSTM recurrent neural network. The image feature vector is first passed through two dense layers whose outputs initialize the hidden and cell states of the LSTM. Text inputs are processed using an embedding layer that maps word indices to dense vector representations. The embedded sequence is then processed by the LSTM, which generates hidden representations for each time step. Dropout regularization is applied to reduce overfitting. Finally, a TimeDistributed Dense layer with softmax activation produces a probability distribution over the vocabulary for predicting the next word.

Model Inputs:
• Image features: Input(shape=(4096,))
• Text sequence: Input(shape=(max_length,))

Model Output:
• Probability distribution over the vocabulary for each word in the sequence: Dense(vocab_size, activation='softmax')

The model was trained in an end-to-end manner to predict the next word in a caption given the image features and previously generated words. A custom data generator was implemented and wrapped using tf.data.Dataset.from_generator for efficient batch processing. The model was optimized using the Adam optimizer and categorical cross-entropy loss function. Training was conducted for 20 epochs while monitoring validation performance to assess generalization and prevent overfitting.

During inference, an image is first processed by the VGG16 encoder to extract its feature representation. These features initialize the hidden and cell states of the trained LSTM decoder. Caption generation begins with the startseq token, and at each step the decoder predicts the most probable next word using greedy search. The generated word is then fed back into the decoder until either the endseq token is produced or the maximum caption length is reached.

Model performance was evaluated using BLEU (Bilingual Evaluation Understudy) scores, which measure the similarity between generated captions and reference captions based on matching n-grams.

The BLEU scores obtained on the test set are presented below:

BLEU-1: 55.36%
BLEU-2: 32.64%
BLEU-3: 20.95%
BLEU-4: 12.75%

The results demonstrate that the proposed VGG16-LSTM architecture is capable of generating meaningful image descriptions and serves as a strong baseline for image captioning. Future work may focus on integrating attention mechanisms or Transformer-based architectures to further improve caption quality and semantic accuracy.
