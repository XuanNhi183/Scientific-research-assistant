/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Paper } from "./types";

export const PRELOADED_PAPERS: Paper[] = [
  {
    id: "attention-is-all-you-need",
    title: "Attention Is All You Need",
    authors: "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin",
    year: "2017",
    journal: "NeurIPS (Neural Information Processing Systems)",
    doi: "10.48550/arXiv.1706.03762",
    abstract: "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.",
    metrics: {
      novelty: 98,
      complexity: "Advanced",
      readingTime: 12,
      citations: 112000
    },
    keyFindings: [
      "Completely eliminates sequential computation (RNN/LSTM), enabling 100% parallel training on GPUs.",
      "Introduces the Multi-Head Self-Attention mechanism to capture relationships between distant words in a sequence.",
      "Establishes a new state-of-the-art BLEU score of 28.4 on the WMT 2014 English-to-German translation task, improving by over 2 BLEU over existing models.",
      "Requires significantly less computational cost and training time than recurrent or convolutional alternatives."
    ],
    sections: [
      {
        title: "1. Introduction",
        content: "Recurrent neural networks, long short-term memory (LSTM) and gated recurrent (GRU) neural networks have been firmly established as state-of-the-art approaches in sequence modeling and transduction problems such as machine translation and language modeling. Due to their sequential nature, the memory or hidden state h(t) depends directly on the previous state h(t-1). This sequential execution prevents parallel training within training examples, which becomes critical at longer sequence lengths. Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, but in almost all cases are combined with a recurrent neural network. In this work, we propose the Transformer - an architecture that bypasses recurrence entirely and relies solely on self-attention."
      },
      {
        title: "2. Model Architecture",
        content: "Most competitive sequence transduction models have an encoder-decoder structure. The encoder maps an input sequence of symbol representations (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). Given z, the decoder then generates an output sequence (y1, ..., ym) of symbols one element at a time.\n\nThe Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder:\n\n- Encoder: Composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection around each of the two sub-layers, followed by layer normalization. The output of each sub-layer is LayerNorm(x + Sublayer(x)).\n\n- Decoder: Also composed of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that predictions for position i can depend only on the known outputs at positions less than i."
      },
      {
        title: "3. Multi-Head Attention",
        content: "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.\n\nWe call our particular attention 'Scaled Dot-Product Attention':\nAttention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V\nwhere d_k is the dimension of the keys. The scaling factor of 1/sqrt(d_k) is crucial, as for large values of d_k, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients.\n\nInstead of performing a single attention function with d_model-dimensional queries, keys and values, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to d_k, d_k and d_v dimensions, respectively. This Multi-Head Attention allows the model to jointly attend to information from different representation subspaces at different positions."
      },
      {
        title: "4. Positional Encoding",
        content: "Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add 'positional encodings' to the input embeddings at the bottoms of both the encoder and decoder stacks. The positional encodings have the same dimension d_model as the embeddings, so that the two can be summed.\n\nIn this work, we use sine and cosine functions of different frequencies:\nPE(pos, 2i) = sin(pos / 10000^(2i/d_model))\nPE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))\nwhere pos is the position and i is the dimension."
      }
    ],
    glossary: [
      {
        term: "Self-Attention",
        definition: "An attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence."
      },
      {
        term: "Multi-Head Attention",
        definition: "A method that splits queries, keys, and values into multiple focus heads, allowing the network to capture multi-faceted token relationships in parallel."
      },
      {
        term: "Layer Normalization",
        definition: "A regularization technique that normalizes inputs across features within a single layer, stabilizing and accelerating deep network training."
      },
      {
        term: "Positional Encoding",
        definition: "A technique that injects physical word positioning or sequential order directly into input embeddings to compensate for the lack of recurrence."
      }
    ]
  },
  {
    id: "resnet-image-recognition",
    title: "Deep Residual Learning for Image Recognition",
    authors: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun",
    year: "2015",
    journal: "CVPR (Computer Vision and Pattern Recognition)",
    doi: "10.1109/CVPR.2016.90",
    abstract: "Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those previously used. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from greatly increased depth.",
    metrics: {
      novelty: 95,
      complexity: "Intermediate",
      readingTime: 10,
      citations: 184000
    },
    keyFindings: [
      "Addresses and resolves the vanishing gradient and accuracy degradation problems caused by stacking too many layers in deep neural architectures.",
      "Introduces 'Skip Connections' or 'Shortcut Connections' which allow information to flow directly across layers without adding parameters or computational complexity.",
      "Successfully trains a 152-layer ResNet, which is 8x deeper than VGG nets while maintaining lower computational complexity.",
      "Achieved first place sweep in all major categories at ILSVRC 2015 and COCO 2015 competitions (ImageNet detection, localization, and COCO segmentation)."
    ],
    sections: [
      {
        title: "1. Introduction",
        content: "Deep convolutional networks have led to a series of breakthroughs for image classification. Network depth is of crucial importance, as depth allows features to be enriched through many stacked layers (low, mid, and high levels). However, a major bottleneck arises when we increase network depth: the accuracy degradation problem. With the network depth increasing, accuracy gets saturated and then degrades rapidly. Unexpectedly, such degradation is not caused by overfitting, and adding more layers to a suitably deep model leads to higher training error. This research addresses this problem by introducing Deep Residual Learning."
      },
      {
        title: "2. Deep Residual Learning",
        content: "Instead of hoping each stack of layers directly fits a desired underlying mapping H(x), we explicitly let these layers fit a residual mapping F(x) := H(x) - x. The original mapping is reformulated into F(x) + x.\n\nWe hypothesize that it is much easier to optimize the residual mapping F(x) than to optimize the original, unreferenced nonlinear mapping H(x). In the extreme case, if an identity mapping was optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers, letting the input flow through unchanged."
      },
      {
        title: "3. Identity Mapping by Shortcuts",
        content: "We adopt residual learning to every few stacked layers. A building block is defined as:\ny = F(x, {W_i}) + x\nwhere x and y are the input and output vectors of the layers. The function F represents the residual mapping to be learned (for instance, 2 or 3 parameterized convolutional layers). The addition '+' is performed through shortcut connections and element-wise addition, which introduces neither extra parameters nor computational complexity.\n\nIf the dimensions of F and x differ, we perform a linear projection W_s via the shortcut to match dimensions:\ny = F(x, {W_i}) + W_s * x"
      }
    ],
    glossary: [
      {
        term: "Skip Connection",
        definition: "A path bypassing one or more intermediate neural layers, adding the original input directly to the layer output."
      },
      {
        term: "Degradation Problem",
        definition: "An optimization failure in extremely deep networks where training error raises as depth increases, distinct from overfitting."
      },
      {
        term: "Identity Mapping",
        definition: "A mathematical transformation that returns its input unaltered: f(x) = x, allowing signals to travel long neural paths without transformation."
      }
    ]
  },
  {
    id: "generative-adversarial-nets",
    title: "Generative Adversarial Nets",
    authors: "Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio",
    year: "2014",
    journal: "NeurIPS (Neural Information Processing Systems)",
    doi: "10.48550/arXiv.1406.2661",
    abstract: "We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game.",
    metrics: {
      novelty: 99,
      complexity: "Advanced",
      readingTime: 11,
      citations: 56000
    },
    keyFindings: [
      "Establishes the tremendously popular field of Generative Adversarial Networks (GANs) in Generative AI.",
      "Formulates a cooperative two-player minimax game that theoretically converges to a unique global optimum where the generator recovers the true data distribution.",
      "Eliminates the need for complex Markov chain sampling or approximate inference calculations during model training.",
      "Lays the foundation for modern ultra-realistic image and video generation, deepfakes, and synthetic style transfers."
    ],
    sections: [
      {
        title: "1. Introduction",
        content: "The promise of deep learning is to discover rich, hierarchical models that represent probability distributions over kinds of data encountered in artificial intelligence. So far, the most striking successes in deep learning have involved discriminative models, usually those that map a high-dimensional, sensory input to a class label. Generative models have had less impact due to the difficulty of approximating many intractable probability calculations and the difficulty of leveraging piecewise linear units with Markov chain sampling. We propose a new framework that sidesteps these bottlenecks by introducing adversarial nets."
      },
      {
        title: "2. Adversarial Nets",
        content: "The minimax game is played over a value function V(D, G):\nmin_G max_D V(D, G) = E_{x~p_data}[log D(x)] + E_{z~p_z}[log (1 - D(G(z)))]\n\nWhere:\n- D(x) represents the probability that x came from the empirical distribution rather than G.\n- G(z) represents the synthetic sample created by the Generator neural network from random input noise z.\n- The objective of the Discriminator (D) is to maximize V, correctly classifying real vs. synthetic data.\n- The objective of the Generator (G) is to minimize V, training itself to fool Discriminator D into believing G(z) is real."
      }
    ],
    glossary: [
      {
        term: "Generator",
        definition: "A neural model that maps low-dimensional latent noise vectors to high-dimensional mock data samples, attempting to simulate the target data distribution."
      },
      {
        term: "Discriminator",
        definition: "A binary classification network that estimates whether an input sample is real (drawn from the empirical dataset) or counterfeit (produced by the generator)."
      },
      {
        term: "Minimax Game",
        definition: "A game-theoretic optimization concept where two agents with opposing utility functions seek to minimize the maximum possible loss."
      }
    ]
  }
];
