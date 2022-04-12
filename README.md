<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/nlpodyssey/spago/main/assets/spago_logo.png" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://github.com/nlpodyssey/spago/workflows/Go/badge.svg?branch=master">
        <img alt="Build" src="https://github.com/nlpodyssey/spago/workflows/Go/badge.svg?branch=master">
    </a>
    <a href="https://codecov.io/gh/nlpodyssey/spago">
        <img alt="Coverage" src="https://codecov.io/gh/nlpodyssey/spago/branch/main/graph/badge.svg">
    </a>
    <a href="https://goreportcard.com/report/github.com/nlpodyssey/spago">
        <img alt="Go Report Card" src="https://goreportcard.com/badge/github.com/nlpodyssey/spago">
    </a>
    <a href="https://codeclimate.com/github/nlpodyssey/spago/maintainability">
        <img alt="Maintainability" src="https://api.codeclimate.com/v1/badges/be7350d3eb1a6a8aa503/maintainability">
    </a>
    <a href="https://pkg.go.dev/github.com/nlpodyssey/spago/">
        <img alt="Documentation" src="https://pkg.go.dev/badge/github.com/nlpodyssey/spago/.svg">
    </a>
    <a href="https://opensource.org/licenses/BSD-2-Clause">
        <img alt="License" src="https://img.shields.io/badge/License-BSD%202--Clause-orange.svg">
    </a>
    <a href="https://github.com/avelino/awesome-go">
        <img alt="Awesome Go" src="https://awesome.re/mentioned-badge.svg">
    </a>
    <a href="http://makeapullrequest.com">
        <img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square">
    </a>
</p>

<p align="center"><i>If you like the project, please ★ star this repository to show your support! 🤩</i></p>

A **Machine Learning** library written in pure Go designed to support relevant neural architectures in **Natural
Language Processing**.

spaGO is self-contained, in that it uses its own lightweight computational graph both for training and
inference, easy to understand from start to finish.

## Features

- Automatic differentiation:
    - Define-by-Run (default, just like PyTorch does)
  
- Optimization methods:
    - Gradient descent (Adam, RAdam, RMS-Prop, AdaGrad, SGD)
  
- Neural networks:
    - Feed-forward models (Linear, Highway, Convolution, ...)
    - Recurrent models (LSTM, GRU, BiLSTM...)
    - Attention mechanisms (Self-Attention, Multi-Head Attention, ...)
    - Recursive auto-encoders

## Usage

Requirements:

* [Go 1.18](https://golang.org/dl/)

Clone this repo or get the library:

```console
go get -u github.com/nlpodyssey/spago
```

At a high level, it comprises four main modules:

1. Matrix
2. Graph
3. Model
4. Optimizer

To get started, look at the implementation of built-in neural models, such as
the [LSTM](https://github.com/nlpodyssey/spago/blob/main/nn/recurrent/lstm/lstm.go). Don't be afraid, it is
straightforward Go code. The idea is that you could have written spaGO :)

## Known Limits

Sadly, at the moment, spaGO is not GPU friendly by design.

## Contributing

We're glad you're thinking about contributing to spaGO! If you think something is missing or could be improved, please
open issues and pull requests. If you'd like to help this project grow, we'd love to have you!

To start contributing, check
the [Contributing Guidelines](https://github.com/nlpodyssey/spago/blob/main/CONTRIBUTING.md).

## Contact

We encourage you to write an issue. This would help the community grow.

If you really want to write to us privately, please email [Matteo Grella](mailto:matteogrella@gmail.com) with your
questions or comments.

## Projects Using spaGO

Below is a list of known projects that use spaGO:

* [GoTransformers](https://github.com/nlpodyssey/gotransformers) - State-of-the-art Natural Language Processing in Go.
* [GoFlair](https://github.com/nlpodyssey/goflair) - State of the art Natural Language Processing in Go.
* [Golem](https://github.com/kirasystems/golem) - A batteries-included implementation
  of ["TabNet: Attentive Interpretable Tabular Learning"](https://arxiv.org/abs/1908.07442).
* [WhatsNew](https://github.com/SpecializedGeneralist/whatsnew/) - A simple tool to collect and process quite a few web
  news from multiple sources.
* [Translator](https://github.com/SpecializedGeneralist/translator) - A simple self-hostable Machine Translation
  service.
* [PySquared](https://github.com/ErikPelli/PiSquared) A Telegram bot that asks you a question and evaluate the response you provide.

## Acknowledgments

spaGO is part of the open-source [NLP Odyssey](https://github.com/nlpodyssey) initiative
initiated by members of the EXOP team (now part of Crisis24). I would therefore like to thank [EXOP GmbH](https://www.exop-group.com/en/) here,
which is providing full support for development by promoting the project and giving it increasing importance.

## Sponsors

We appreciate contributions of all kinds. We especially want to thank spaGO fiscal sponsors who contribute to ongoing
project maintenance.

See our [Open Collective](https://opencollective.com/nlpodyssey/contribute) page if you too are interested in becoming a sponsor.
