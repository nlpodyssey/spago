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

<div style="text-align:center"><img src="https://github.com/nlpodyssey/spago/blob/main/assets/screenshot_spago_api_qa.png" /></div>

**Is spaGO right for me?**

Are you looking for a highly optimized, scalable, battle-tested, production-ready machine-learning/NLP framework? Are you also a Python lover and enjoy manipulating tensors? If yes, you won't find much to your satisfaction here. [PyTorch](https://pytorch.org/) plus the wonders of the friends of [Hugging Face](https://github.com/huggingface) is the answer you seek!

If instead you prefer statically typed, compiled programming language, and a **simpler yet well-structured** machine-learning framework almost ready to use is what you need, then you are in the right place!

The idea is that you could have written spaGO. Most of it, from the computational graph to the [LSTM](https://github.com/nlpodyssey/spago/blob/main/pkg/ml/nn/rec/lstm/lstm.go#L182) is straightforward Go code :)

## Features

* Automatic differentiation
    - You write the *forward()*, it does all *backward()* derivatives for you:
        -   Define-by-Run (default, just like PyTorch does)
        -   Define-and-Run (similar to the static graph of TensorFlow)
* Optimization methods
    - Adam, RAdam, RMS-Prop, AdaGrad, SGD
* Neural networks
    -   Feed-forward models (Linear, Highway, Convolution, ...)
    -   Recurrent models (LSTM, GRU, BiLSTM...)
    -   Attention mechanisms (Self-Attention, Multi-Head Attention, ...)
    -   Recursive auto-encoders
* Natural Language Processing
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
* Compatible with pre-trained state-of-the-art neural models:
    - 🤗 BERT-like [Transformers](https://github.com/huggingface/transformers)
    - [Flair](https://github.com/flairNLP/flair) sequence labeler architecture

## Documentation

* [Godocs for spaGo](https://pkg.go.dev/mod/github.com/nlpodyssey/spago)
* [Contributing to spaGo](CONTRIBUTING.md)

### Demos

Several demo programs can be leveraged to tour the current capabilities in spaGO. The demos are documented on the [Wiki](https://github.com/nlpodyssey/spago/wiki/Demos). A list of the demos now follows.

* [Named Entities Recognition](https://github.com/nlpodyssey/spago/wiki/Demos#named-entities-recognition-demo)
* [Import a Pre-Trained Model](https://github.com/nlpodyssey/spago/wiki/Demos#import-a-pre-trained-model-demo)
* [Question Answering](https://github.com/nlpodyssey/spago/wiki/Demos#question-answering-demo)
* [Masked Language Model](https://github.com/nlpodyssey/spago/wiki/Demos#masked-language-model-demo)

### Feature Source Tree

A tree-like view of the currently supported features in the library now follows.

```bash
pkg
├── mat
│   ├── matrix.go
│   ├── dense.go
│   ├── sparse.go
│   └── rand
│       ├── bernulli
│       ├── normal
│       └── uniform
└── ml (machine learning)
│   ├── ag (auto-grad)
│   │   ├── fn (functions with automatic differentiation)
│   │   │   ├── add.go
│   │   │   ├── at.go
│   │   │   ├── concat.go
│   │   │   ├── div.go
│   │   │   ├── dot.go
│   │   │   ├── dropout.go
│   │   │   ├── elu.go
│   │   │   ├── fn.go
│   │   │   ├── identity.go
│   │   │   ├── leakyrelu.go
│   │   │   ├── maxpooling.go
│   │   │   ├── misc.go
│   │   │   ├── mul.go
│   │   │   ├── pow.go
│   │   │   ├── prod.go
│   │   │   ├── reducemean.go
│   │   │   ├── reducesum.go
│   │   │   ├── reshape.go
│   │   │   ├── softmax.go
│   │   │   ├── stack.go
│   │   │   ├── sub.go
│   │   │   ├── subscalar.go
│   │   │   ├── swish.go
│   │   │   ├── swish_test.go
│   │   │   ├── threshold.go
│   │   │   ├── transpose.go
│   │   │   ├── unaryelementwise.go
│   │   │   ├── ...
│   │   ├── gradvalue.go
│   │   ├── graph.go (computational graph)
│   │   ├── node.go
│   │   ├── operator.go
│   │   ├── operators.go
│   │   ├── variable.go
│   │   └── wrapper.go
│   ├── encoding
│   │   ├── fofe
│   │   │   ├── decoder.go
│   │   │   ├── encoder.go
│   │   └── pe (positional encoding)
│   │       └── encoder.go
│   ├── initializers
│   │   ├── Constant
│   │   ├── Uniform
│   │   ├── Normal
│   │   ├── Xavier (Glorot)
│   ├── losses
│   │   ├── MAE
│   │   ├── MSE
│   │   ├── NLL
│   │   ├── CrossEntropy
│   ├── nn
│   │   ├── model.go (neural model and neural processor interfaces)
│   │   ├── transforms.go (e.g. Affine, Conv2D, Self-Attention)
│   │   ├── param.go (weights, biases)
│   │   ├── activation
│   │   ├── birnn (bi-directional recurrent neural network)
│   │   ├── bls (broad learning system)
│   │   ├── cnn
│   │   ├── convolution
│   │   ├── crf
│   │   ├── highway
│   │   ├── selfattention
│   │   ├── syntheticattention
│   │   ├── multiheadattention
│   │   ├── normalization
│   │   │   ├── adanorm
│   │   │   ├── batchnorm
│   │   │   ├── fixnorm
│   │   │   ├── layernorm
│   │   │   ├── layernormsimple
│   │   │   ├── rmsnorm
│   │   │   └── scalenorm
│   │   ├── linear
│   │   ├── rae (recursive auto-encoder)
│   │   ├── rec (recurrent models)
│   │   │   ├── cfn
│   │   │   ├── deltarnn
│   │   │   ├── fsmn
│   │   │   ├── gru
│   │   │   ├── horn
│   │   │   ├── indrnn
│   │   │   ├── lstm
│   │   │   ├── lstmsc
│   │   │   ├── ltm
│   │   │   ├── mist
│   │   │   ├── nru
│   │   │   ├── ran
│   │   │   ├── srn
│   │   │   └── tpr
│   │   ├── sqrdist
│   │   └── stack
│   └── optimizers
│       ├── de (differential evolution)
│       │   ├── de.go
│       │   ├── crossover.go
│       │   ├── member.go
│       │   ├── mutator.go
│       │   └── population.go
│       ├── gd (gradient descent)
│       │   ├── sgd
│       │   ├── rmsprop
│       │   ├── adagrad
│       │   ├── adam
│       │   ├── radam
│       │   ├── clipper
│       │   ├── decay
│       │   │   ├── exponential
│       │   │   └── hyperbolic
│       │   ├── gd.go
│       │   └── scheduler.go
│       └── optimizer.go (interface implemented by all optimizers)
└── nlp (natural language processing)
    ├── embeddings
    ├── contextual string embeddings
    ├── evolving embeddings
    ├── charlm (characters language model)
    ├── sequence labeler
    ├── tokenizers
    │   ├── base (whitespaces and punctuation)
    │   └── wordpiece
    ├── transformer (BERT-like models)
    ├── vocabulary
    └── corpora
```

Please note that the structure above does not reflect the original folder structure (although it is very close). I added comments and deleted files to keep the visualization compact.

The inclusion of neural models in the **nn** sub-package is mostly arbitrary. Not all neural models are useful. For instance, I wanted to implement many recurrent networks for the sake of curiosity, but in the end, the LSTM and GRU almost always gave me the best performance in natural language processing tasks (from language modelling to syntactic parsing). I might decide - based on your suggestions - to delete some of them to lighten the core package. 

## Requirements

* [Go 1.14](https://golang.org/dl/)
* [go modules](https://blog.golang.org/using-go-modules)

## Usage

Get the library:

```console
go get -u github.com/nlpodyssey/spago
```

The [ag package](https://github.com/nlpodyssey/spago/tree/main/pkg/ml/ag) (a.k.a. auto-grad) is the centerpiece of the spaGO machine learning framework.

Neural models optimized by back-propagation require gradients to be available during training.
The set of expressions characterizing the forward-step of such models must be defined within the [ag.Graph](https://github.com/nlpodyssey/spago/blob/main/pkg/ml/ag/graph.go) to take advantage of automatic differentiation.

Let's see if spaGO can tell us what two plus five is.
Then, let's go one step further now and ask spaGO to give us the gradients on `a` and `b`, starting with arbitrary output gradients.

Write some code:

```go
package main

import (
	"fmt"

	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
)

func main() {
	// create a new node of type variable with a scalar
	a := ag.NewVariable(mat.NewScalar(2.0), true)
	// create another node of type variable with a scalar
	b := ag.NewVariable(mat.NewScalar(5.0), true)
	// create an addition operator (the calculation is actually performed here)
	c := ag.Add(a, b)
	// print the result
	fmt.Printf("c = %v\n", c.Value())

	ag.Backward(c, ag.OutputGrad(mat.NewScalar(0.5)))
	fmt.Printf("ga = %v\n", a.Grad())
	fmt.Printf("gb = %v\n", b.Grad())
}
```

It should print:

```console
c = [7]
ga = [0.5]
gb = [0.5]
```

You will soon find some tutorials on the [Wiki](https://github.com/nlpodyssey/spago/wiki/Machine-Learning-Framework).

## Current Status
We're not at a v1.0.0 yet, so spaGO is currently an experimental work-in-progress. 
It's pretty easy to get your hands on through, so you might want to use it in your real applications. Early adopters may make use of it for production use today as long as they understand and accept that spaGO is not fully tested and that APIs will change (maybe extensively).

If you're wondering, I haven't used spaGO in production yet, but I plan to do the first integration tests soon.

## Project Goals

### Why spaGO?

I've been writing more or less the same software for almost 20 years. I guess it's my way of learning a new language. Now it's Go's turn, and spaGO is the result of a few days of pure fun!

Let me explain a little further. It's not precisely the very same software I've been writing now for 20 years: I've been working in the NLP for this long, experimenting with different approaches and techniques, and therefore software of the same field. 
I've always felt satisfied to limit the use of third-party dependencies, writing firsthand the algorithms that interest me most. 
So, I took the opportunity to speed up my understanding of the deep learning techniques and methodologies underlying cutting-edge NLP results, implementing them almost from scratch in straightforward Go code.
I'm aware that [reinventing the wheel](https://en.wikipedia.org/wiki/Reinventing_the_wheel#Related_phrases) is an anti-pattern; nevertheless, I wanted to build something with my own concepts in my own (italian) style: that's the way I learn best, and it could be your best chance to understand what's going on under the hood of the artificial intelligence :)

When I start programming in a new language, I usually do not know much of it. I often combine the techniques I have acquired by writing in other languages and other paradigms, so some choices may not be the most idiomatic... but who cares, right? 

It's with this approach that I jumped on Go and created spaGo: a work in progress, (hopefully) understandable, easy to use library for machine learning and natural language processing.

### What direction did you take for the development of spaGO?

I started spaGO to deepen first-hand the mechanisms underlying a machine learning framework. In doing this, I thought it was an excellent opportunity to set up the library so to enable the use and understanding of such algorithms to non-experts as well. 

In my experience, the first barrier to (deep) machine learning for developers who do not enjoy mathematics, at least not too much, is getting familiar with the use of tensors rather than understanding neural architecture. Well, in spaGO, we only use well-known 2D Matrices, by which we can represent vectors and scalars too. That's all we need (performance aside). You won't lose sleep anymore by watching tensor axes to figure out how to do math operations. 

Since it's a counter-trend decision, let me argue some more. It happened a few times that friends and colleagues, who are super cool full-stack developers, tried to understand the NLP algorithms I was programming in PyTorch. Sometimes they gave up just because "the forward() method doesn't look like the usual code" to them. 

Honestly, I don't find it hard to believe that by combining Python's dynamism with the versatility of tensors, the flow of a program can become hard to digest. It is undoubtedly essential to devote a good time reading the documentation, which may not be immediately available. Hence, you find yourself forced to inspect the content of the variables at runtime with your favorite IDE (PyCharm, of course). It happens in general, but I believe in machine learning in particular.

In other words, I wanted to limit as much as possible the use of tensors larger than two dimensions, preferring the use of built-in types such as slices and maps. For example, batches are explicit as slices of nodes, not part of the same forward() computation. Too much detail here, sorry. At the end, I guess we do gain static code analysis this way, by shifting the focus from the tensor operations back to traditional control-flows. Of course, the type checker still can't verify the correct shapes of matrices and the like. That still requires runtime panics etc. I agree that it is hard to see where to draw the line, but so far, I'm pretty happy with my decision.

### Caveat

Sadly, not using tensors, spaGO is not GPU or TPU friendly by design. You bet, I'm going to do some experiments integrating CUDA, but I can already tell you that I will not reach satisfactory levels.

In spaGO, using slices of (slices of) matrices, we have to "loop" often to do mathematical operations, whereas they are performed in one go using tensors. Any time your code has a loop that is not GPU or TPU friendly.  

Mainstream machine-learning tensor-based frameworks such as PyTorch and TensorFlow, the first thing they want to do, is to convert whatever you're doing into a big matrix multiplication problem, which is where the GPU does its best. Yeah, that's an overstatement, but not so far from reality. Storing all data in tensors and applying batched operations to them is the way to go for hardware acceleration. On GPU, it's a must, and even on CPU, that could give a 10x speedup or more with cache-aware BLAS libraries.

Beyond that, I think there's a lot of basic design improvements that would be necessary before spaGO could fit for mainstream use. Many boilerplates could go away using reflection, or more simply by careful engineering. It's perfectly normal; the more I program in Go, the more I would review some choices.

## Acknowledgments

spaGO is a personal project that is part of the open-source [NLP Odyssey](https://github.com/nlpodyssey) initiative initiated by members of the EXOP team. I would therefore like to thank [EXOP GmbH](https://www.exop-group.com/en/) here, which is providing full support for development by promoting the project and giving it increasing importance.

## Contact

I encourage you to write an issue. This would help the community grow.

If you really want to write to me privately, please email [Matteo Grella](mailto:matteogrella@gmail.com) with your questions or comments.
