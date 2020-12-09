![alt text](https://github.com/nlpodyssey/spago/blob/main/assets/spago_logo.png)

![Go](https://github.com/nlpodyssey/spago/workflows/Go/badge.svg?branch=master)
[![Go Report Card](https://goreportcard.com/badge/github.com/nlpodyssey/spago)](https://goreportcard.com/report/github.com/nlpodyssey/spago)
[![Maintainability](https://api.codeclimate.com/v1/badges/be7350d3eb1a6a8aa503/maintainability)](https://codeclimate.com/github/nlpodyssey/spago/maintainability)
[![codecov](https://codecov.io/gh/nlpodyssey/spago/branch/main/graph/badge.svg)](https://codecov.io/gh/nlpodyssey/spago)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
![Unstable](https://github.com/nlpodyssey/spago/blob/main/assets/stability-unstable-yellow.svg)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

<p align="center"><i>If you like the project, please ★ star this repository to show your support! 🤩</i></p>

A beautiful and maintainable machine learning library written in Go. It is designed to support relevant neural architectures in **Natural Language Processing**.

spaGO is compatible with 🤗 BERT-like [Transformers](https://github.com/huggingface/transformers) and with the [Flair](https://github.com/flairNLP/flair) sequence labeler architecture. 

## Features

#### Automatic differentiation
- You write the *forward()*, it does all *backward()* derivatives for you:
    -   Define-by-Run (default, just like PyTorch does)
    -   Define-and-Run (similar to the static graph of TensorFlow)

#### Optimization methods
- Gradient descent:
    - Adam, RAdam, RMS-Prop, AdaGrad, SGD
- Differential Evolution

#### Neural networks
-   Feed-forward models (Linear, Highway, Convolution, ...)
-   Recurrent models (LSTM, GRU, BiLSTM...)
-   Attention mechanisms (Self-Attention, Multi-Head Attention, ...)
-   Recursive auto-encoders

#### Natural Language Processing
-   Memory-efficient Word Embeddings (with [badger](https://github.com/dgraph-io/badger) key–value store)
-   Character Language Models
-   Recurrent Sequence Labeler with CRF on top (e.g. Named Entities Recognition)
-   Transformer models (BERT-like)
    -   Masked language model
    -   Next sentence prediction
    -   Tokens Classification
    -   Text Classification (e.g. Sentiment Analysis)
    -   Question Answering
    -   Textual Entailment
    -   Text Similarity

#### Compatible with pre-trained state-of-the-art neural models:
- 🤗 BERT-like [Transformers](https://github.com/huggingface/transformers)
- [Flair](https://github.com/flairNLP/flair) sequence labeler architecture

## Documentation

* [Godocs for spaGO](https://pkg.go.dev/mod/github.com/nlpodyssey/spago)
* [Feature Source Tree](https://github.com/nlpodyssey/spago/wiki/Feature-Source-Tree)
* [Contributing to spaGO](CONTRIBUTING.md)
* [spaGO: Self-contained ML/NLP Library in Go](https://www.slideshare.net/MatteoGrella/spago-a-selfcontained-ml-nlp-library-in-go) ([video of GoWayFest 4.0](https://www.youtube.com/watch?v=wE3CQU4G2fk))

## Usage

Requirements:

* [Go 1.15](https://golang.org/dl/)
* [Go modules](https://blog.golang.org/using-go-modules)

Clone this repo or get the library:

```console
go get -u github.com/nlpodyssey/spago
```

To get started, you can find some tutorials on the [Wiki](https://github.com/nlpodyssey/spago/wiki) about the [Machine Learning Framework](https://github.com/nlpodyssey/spago/wiki/Machine-Learning-Framework).

Several demo programs can be leveraged to tour the current capabilities in spaGO. The demos are documented on this [page](https://github.com/nlpodyssey/spago/wiki/Demos) of the Wiki. A list of the demos now follows.

* [Named Entities Recognition](https://github.com/nlpodyssey/spago/wiki/Demos#named-entities-recognition-demo)
* [Import a Pre-Trained Model](https://github.com/nlpodyssey/spago/wiki/Demos#import-a-pre-trained-model-demo)
* [Question Answering](https://github.com/nlpodyssey/spago/wiki/Demos#question-answering-demo)
* [Masked Language Model](https://github.com/nlpodyssey/spago/wiki/Demos#masked-language-model-demo)

There is also a [repo](https://github.com/nlpodyssey/spago-examples) with handy examples, such as MNIST classification.

## Project Goals

<details><summary><b>Is spaGO right for me?</b></summary>
<p>

Are you looking for a highly optimized, scalable, battle-tested, production-ready machine-learning/NLP framework? Are you also a Python lover and enjoy manipulating tensors? If yes, you won't find much to your satisfaction here.

[PyTorch](https://pytorch.org/) plus the wonders of the friends of [Hugging Face](https://github.com/huggingface) is the answer you seek!

If instead you prefer statically typed, compiled programming language, and a **simpler yet well-structured** machine-learning framework almost ready to use is what you need, then you are in the right place!

The idea is that you could have written spaGO. Most of it, from the computational graph to the [LSTM](https://github.com/nlpodyssey/spago/blob/main/pkg/ml/nn/rec/lstm/lstm.go#L182) is straightforward Go code :)
</p>
</details>

<details><summary><b>Why spaGO?</b></summary>
<p>

I've been writing more or less the same software for almost 20 years. I guess it's my way of learning a new language. Now it's Go's turn, and spaGO is the result of a few days of pure fun!

Let me explain a little further. It's not precisely the very same software I've been writing now for 20 years: I've been working in the NLP for this long, experimenting with different approaches and techniques, and therefore software of the same field. 
I've always felt satisfied to limit the use of third-party dependencies, writing firsthand the algorithms that interest me most. 
So, I took the opportunity to speed up my understanding of the deep learning techniques and methodologies underlying cutting-edge NLP results, implementing them almost from scratch in straightforward Go code.
I'm aware that [reinventing the wheel](https://en.wikipedia.org/wiki/Reinventing_the_wheel#Related_phrases) is an anti-pattern; nevertheless, I wanted to build something with my own concepts in my own (italian) style: that's the way I learn best, and it could be your best chance to understand what's going on under the hood of the artificial intelligence :)

When I start programming in a new language, I usually do not know much of it. I often combine the techniques I have acquired by writing in other languages and other paradigms, so some choices may not be the most idiomatic... but who cares, right? 

It's with this approach that I jumped on Go and created spaGo: a work in progress, (hopefully) understandable, easy to use library for machine learning and natural language processing.
</p>
</details>

<details><summary><b>What direction did you take for the development of spaGO?</b></summary>
<p>

I started spaGO to deepen first-hand the mechanisms underlying a machine learning framework. In doing this, I thought it was an excellent opportunity to set up the library so to enable the use and understanding of such algorithms to non-experts as well. 

In my experience, the first barrier to (deep) machine learning for developers who do not enjoy mathematics, at least not too much, is getting familiar with the use of tensors rather than understanding neural architecture. Well, in spaGO, we only use well-known 2D Matrices, by which we can represent vectors and scalars too. That's all we need (performance aside). You won't lose sleep anymore by watching tensor axes to figure out how to do math operations. 

Since it's a counter-trend decision, let me argue some more. It happened a few times that friends and colleagues, who are super cool full-stack developers, tried to understand the NLP algorithms I was programming in PyTorch. Sometimes they gave up just because "the forward() method doesn't look like the usual code" to them. 

Honestly, I don't find it hard to believe that by combining Python's dynamism with the versatility of tensors, the flow of a program can become hard to digest. It is undoubtedly essential to devote a good time reading the documentation, which may not be immediately available. Hence, you find yourself forced to inspect the content of the variables at runtime with your favorite IDE (PyCharm, of course). It happens in general, but I believe in machine learning in particular.

In other words, I wanted to limit as much as possible the use of tensors larger than two dimensions, preferring the use of built-in types such as slices and maps. For example, batches are explicit as slices of nodes, not part of the same forward() computation. Too much detail here, sorry. At the end, I guess we do gain static code analysis this way, by shifting the focus from the tensor operations back to traditional control-flows. Of course, the type checker still can't verify the correct shapes of matrices and the like. That still requires runtime panics etc. I agree that it is hard to see where to draw the line, but so far, I'm pretty happy with my decision.
</p>
</details>

<details><summary><b>Does spaGO support GPU?</b></summary>
<p>

Sadly, not using tensors, spaGO is not GPU or TPU friendly by design. You bet, I'm going to do some experiments integrating CUDA, but I can already tell you that I will not reach satisfactory levels.

In spaGO, using slices of (slices of) matrices, we have to "loop" often to do mathematical operations, whereas they are performed in one go using tensors. Any time your code has a loop that is not GPU or TPU friendly.  

Mainstream machine-learning tensor-based frameworks such as PyTorch and TensorFlow, the first thing they want to do, is to convert whatever you're doing into a big matrix multiplication problem, which is where the GPU does its best. Yeah, that's an overstatement, but not so far from reality. Storing all data in tensors and applying batched operations to them is the way to go for hardware acceleration. On GPU, it's a must, and even on CPU, that could give a 10x speedup or more with cache-aware BLAS libraries.

Beyond that, I think there's a lot of basic design improvements that would be necessary before spaGO could fit for mainstream use. Many boilerplates could go away using reflection, or more simply by careful engineering. It's perfectly normal; the more I program in Go, the more I would review some choices.
</p>
</details>

<details><summary><b>Is spaGO stable?</b></summary>
<p>

We're not at a v1.0.0 yet, so spaGO is currently an experimental work-in-progress. 
It's pretty easy to get your hands on through, so you might want to use it in your real applications. Early adopters may make use of it for production use today as long as they understand and accept that spaGO is not fully tested and that APIs will change (maybe extensively).

<s>If you're wondering, I haven't used spaGO in production yet, but I plan to do the first integration tests soon.</s>

spaGO has been running smoothly for a couple of months now in a system that analyzes thousands of news items a day!
</p>
</details>

## Contact

I encourage you to write an issue. This would help the community grow.

If you really want to write to me privately, please email [Matteo Grella](mailto:matteogrella@gmail.com) with your questions or comments.

## Acknowledgments

spaGO is a personal project that is part of the open-source [NLP Odyssey](https://github.com/nlpodyssey) initiative initiated by members of the EXOP team. I would therefore like to thank [EXOP GmbH](https://www.exop-group.com/en/) here, which is providing full support for development by promoting the project and giving it increasing importance.

## Sponsors

I appreciate contributions of all kinds. I especially want to thank spaGO fiscal sponsors who contribute to ongoing project maintenance.

![Faire.ai logo](https://github.com/nlpodyssey/spago/blob/main/assets/sponsors/faire_ai_logo.png)

> Our aim is simplifying people's life by making lending easy, fast and accessible, leveraging Open Banking and Artificial Intelligence.
> https://www.faire.ai/

See our [Open Collective](https://opencollective.com/nlpodyssey/contribute) page if you too are interested in becoming a sponsor.