![alt text](https://raw.githubusercontent.com/nlpodyssey/spago/main/assets/spago_logo.png)

![Go](https://github.com/nlpodyssey/spago/workflows/Go/badge.svg?branch=master)
[![Go Report Card](https://goreportcard.com/badge/github.com/nlpodyssey/spago)](https://goreportcard.com/report/github.com/nlpodyssey/spago)
[![Maintainability](https://api.codeclimate.com/v1/badges/be7350d3eb1a6a8aa503/maintainability)](https://codeclimate.com/github/nlpodyssey/spago/maintainability)
[![codecov](https://codecov.io/gh/nlpodyssey/spago/branch/main/graph/badge.svg)](https://codecov.io/gh/nlpodyssey/spago)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
![Unstable](https://raw.githubusercontent.com/nlpodyssey/spago/main/assets/stability-unstable-yellow.svg)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

<p align="center"><i>If you like the project, please ★ star this repository to show your support! 🤩</i></p>

A **Machine Learning** library written in pure Go designed to support relevant neural architectures in **Natural
Language Processing**.

spaGO is self-contained, in that it uses its own lightweight *computational graph* framework for both training and
inference, easy to understand from start to finish.

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

- Hugging Face [Transformers](https://github.com/huggingface/transformers)
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

## Current Status

We're not at a v1.0.0 yet, so spaGO is currently work-in-progress.

However, it has been running smoothly for a quite a few months now in a system that analyzes thousands of news items a
day!
Besides, it's pretty easy to get your hands on through, so you might want to use it in your real applications.

Early adopters may make use of it for production use today as long as they understand and accept that spaGO is not fully
tested and that APIs might change.

## Known Limits

Sadly, at the moment, spaGO is not GPU friendly by design.

## Contact

We encourage you to write an issue. This would help the community grow.

If you really want to write to us privately, please email [Matteo Grella](mailto:matteogrella@gmail.com) with your
questions or comments.

## Acknowledgments

spaGO is a personal project that is part of the open-source [NLP Odyssey](https://github.com/nlpodyssey) initiative initiated by members of the EXOP team. I would therefore like to thank [EXOP GmbH](https://www.exop-group.com/en/) here, which is providing full support for development by promoting the project and giving it increasing importance.

## Sponsors

We appreciate contributions of all kinds. We especially want to thank spaGO fiscal sponsors who contribute to ongoing
project maintenance.

![Faire.ai logo](https://raw.githubusercontent.com/nlpodyssey/spago/main/assets/sponsors/faire_ai_logo.png)

> Our aim is simplifying people's life by making lending easy, fast and accessible, leveraging Open Banking and Artificial Intelligence.
> https://www.faire.ai/

![Hype logo](https://raw.githubusercontent.com/nlpodyssey/spago/main/assets/sponsors/hype_design_logo.png)

> We work on Artificial Intelligence based hardware and software systems, declining them in areas such as Energy Management, Personal Safety, E-Health and Sports equipment.
> https://hype-design.it/

![BoxxApps logo](https://raw.githubusercontent.com/nlpodyssey/spago/main/assets/sponsors/boxxapps_logo.png)

> Professional services in the IT sector for Local Administrations, Enterprises and Local Authorities.
> https://www.boxxapps.com/

See our [Open Collective](https://opencollective.com/nlpodyssey/contribute) page if you too are interested in becoming a sponsor.
