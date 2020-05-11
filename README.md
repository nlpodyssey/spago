spaGO
=====
[![Go Report Card](https://goreportcard.com/badge/github.com/nlpodyssey/spago)](https://goreportcard.com/report/github.com/nlpodyssey/spago)

spaGO is a machine learning lightweight open-source library written in Go designed to support relevant neural network architectures in natural language processing tasks.

Are you looking for a highly optimized, scalable, battle-tested, production-ready machine-learning/NLP framework? Are you also a Python lover and enjoy manipulating tensors? If yes, you won't find much to your satisfaction here. [PyTorch](https://pytorch.org/) plus the wonders of the friends of [Hugging Face](https://github.com/huggingface) is the answer you seek!

If instead you prefer statically typed, compiled programming language, and a **simpler yet well-structured** machine-learning framework almost ready to use is what you need, then you are in the right place!
The idea is that you could have written spaGO. Most of it, from the computational graph to the [LSTM](https://github.com/nlpodyssey/spago/blob/master/pkg/ml/nn/rec/lstm/lstm.go#L182) is straightforward Go code :)

### Note

**This README is under construction, you might consider coming back here in a few days, or go straight to the code if you're curious!**

Installation
=====
Make sure you have [Go 1.14](https://golang.org/dl/) installed on your computer first. The package can be installed using *go get* as follows:

```console
go get -u https://github.com/nlpodyssey/spago
```

spaGO is compatible with [go modules](https://blog.golang.org/using-go-modules).

What's inside?
=====

spaGO ships with a ton of built-in features, including:
- Automatic differentiation (you write the *forward()*, it does all *backward()* derivatives for you):
    -   Define-by-Run (default)
    -   Define-and-Run
- Feed-forward neural networks
- Recurrent neural networks
- Transformer neural networks

I haven't found the time yet to write a proper documentation, or at least a clear description of what spaGO contains.

To start with, I thought that a tree-like view of the core contents of the library (*pkg* folder) might help you to understand the current supported features, and - more important - how I decided to structure spaGO in the first place.

The names I have adopted for the various sub-packages and files should be self-explanatory enough. Well, at least that was my intention during development :)

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
    ├── ag (auto-grad)
    │   ├── fn (functions with automatic differentiation)
    │   │   ├── add.go
    │   │   ├── at.go
    │   │   ├── concat.go
    │   │   ├── div.go
    │   │   ├── dot.go
    │   │   ├── dropout.go
    │   │   ├── elu.go
    │   │   ├── fn.go
    │   │   ├── identity.go
    │   │   ├── leakyrelu.go
    │   │   ├── maxpooling.go
    │   │   ├── misc.go
    │   │   ├── mul.go
    │   │   ├── pow.go
    │   │   ├── prod.go
    │   │   ├── reducemean.go
    │   │   ├── reducesum.go
    │   │   ├── reshape.go
    │   │   ├── softmax.go
    │   │   ├── stack.go
    │   │   ├── sub.go
    │   │   ├── subscalar.go
    │   │   ├── swish.go
    │   │   ├── swish_test.go
    │   │   ├── threshold.go
    │   │   ├── transpose.go
    │   │   ├── unaryelementwise.go
    │   │   ├── ...
    │   ├── gradvalue.go
    │   ├── graph.go (computational graph)
    │   ├── node.go
    │   ├── operator.go
    │   ├── operators.go
    │   ├── variable.go
    │   └── wrapper.go
    ├── emb
    │   ├── embedding.go
    │   └── embmap.go
    ├── encoding
    │   ├── fofe
    │   │   ├── decoder.go
    │   │   ├── encoder.go
    │   └── pe (positional encoding)
    │       └── encoder.go
    ├── initializers
    │   ├── Constant
    │   ├── Uniform
    │   ├── Normal
    │   ├── Xavier (Glorot)
    ├── losses
    │   ├── MAE
    │   ├── MSE
    │   ├── NLL
    │   ├── CrossEntropy
    ├── nn
    │   ├── model.go (neural model and neural processor interfaces)
    │   ├── transforms.go (e.g. Affine, Conv2D, Self-Attention)
    │   ├── param.go (weights, biases)
    │   ├── activation
    │   ├── birnn (bi-directional recurrent neural network)
    │   ├── bls (broad learning system)
    │   ├── cnn
    │   ├── convolution
    │   ├── crf
    │   ├── highway
    │   ├── selfattention
    │   ├── multiheadattention
    │   ├── normalization
    │   │   ├── adanorm
    │   │   ├── batchnorm
    │   │   ├── fixnorm
    │   │   ├── layernorm
    │   │   ├── layernormsimple
    │   │   ├── rmsnorm
    │   │   └── scalenorm
    │   ├── linear
    │   ├── rae (recursive auto-encoder)
    │   ├── rec (recurrent models)
    │   │   ├── cfn
    │   │   ├── deltarnn
    │   │   ├── fsmn
    │   │   ├── gru
    │   │   ├── horn
    │   │   ├── indrnn
    │   │   ├── lstm
    │   │   ├── lstmsc
    │   │   ├── ltm
    │   │   ├── mist
    │   │   ├── nru
    │   │   ├── ran
    │   │   ├── srn
    │   │   └── tpr
    │   ├── sqrdist
    │   ├── stack
    │   ├── transformer (BERT-like model)
    └── optimizers
        ├── de (differential evolution)
        │   ├── de.go
        │   ├── crossover.go
        │   ├── member.go
        │   ├── mutator.go
        │   └── population.go
        ├── gd (gradient descent)
        │   ├── sgd
        │   ├── rmsprop
        │   ├── adagrad
        │   ├── adam
        │   ├── clipper
        │   ├── decay
        │   │   ├── exponential
        │   │   └── hyperbolic
        │   ├── gd.go
        │   └── scheduler.go
        └── optimizer.go (interface implemented by all optimizers)
```

Please note that the structure above does not reflect the original folder structure (although it is very close). I added comments and deleted files to keep the visualization compact.

### Note

The inclusion of neural models in the **nn** sub-package is mostly arbitrary. Not all neural models are useful. I might decide - based on your suggestions - to delete some of them to lighten the core package of the library. For instance, I wanted to implement many recurrent networks for the sake of curiosity, but in the end, the LSTM and GRU almost always gave me the best performance in natural language processing tasks (from language modelling to syntactic parsing).

Current status
=====
We're not at a v1.0.0 yet, so spaGO is currently an experimental work-in-progress. 
It's pretty easy to get your hands on, so you might want to use it in your real applications. Early adopters may make use of it for production use today as long as they understand and accept that spaGO is not fully tested and that APIs will change (maybe extensively).

> If you're wondering, I haven't used spaGO in production yet, but I plan to do the first integration tests soon.

Design choices
=====

I'm trying to write efficient, beautiful, and maintainable code. I said try, not that I can do it ;) I'm sure you know the feeling of being happy merely looking at the lines of code you just wrote. Well, I'm doing my best to develop spaGO in a way that makes me always feel this way. 

I want to keep the code self-documenting and straightforward, starting with the organization of the packages. By that, I don't mean that I don't have to improve the documentation tremendously. It's my very next step.

I started spaGO to deepen first-hand the mechanisms underlying a machine learning framework. In doing this, I thought it was an excellent opportunity to set up the library so to enable the use and understanding of such algorithms to non-experts as well. 

In my experience, the first barrier to (deep) machine learning for developers who do not enjoy mathematics, at least not too much, is getting familiar with the use of tensors rather than understanding neural architecture. Well, there are no tensors in spaGO, only well-known 2D Matrices, by which we can represent vectors and scalars too. That's all we need (performance aside). You won't lose sleep anymore by watching tensor axes to figure out how to do math operations. 

Since it's a counter-trend decision, let me argue some more. It happened a few times that friends and colleagues, who are super cool full-stack developers, tried to understand the NLP algorithms I was programming in PyTorch. Sometimes they gave up just because "the forward() method doesn't look like the usual code" to them. 

Honestly, I don't find it hard to believe that by combining Python's dynamism with the versatility of tensors, the flow of a program can become hard to digest. It is undoubtedly essential to devote a good time reading the documentation, which may not be immediately available. Hence, you find yourself forced to inspect the content of the variables at runtime with your favorite IDE (PyCharm, of course). It happens in general, but I believe in machine learning in particular.

In other words, I wanted to limit as much as possible the use of tensors larger than two dimensions, preferring the use of built-in types such as slices and maps. For example, batches are explicit as slices of nodes, not part of the same forward() computation. Too much detail here, sorry. At the end, I guess we do gain static code analysis this way, by shifting the focus from the tensor operations back to traditional control-flows. Of course, the type checker still can't verify the correct shapes of matrices and the like. That still requires runtime panics etc. I agree that it is hard to see where to draw the line, but so far, I'm pretty happy with my decision.

### Caveat

Sadly, not using tensors, spaGO is not GPU or TPU friendly by design. You bet, I'm going to do some experiments integrating CUDA, but I can already tell you that I will not reach satisfactory levels.

In spaGO, using slices of (slices of) matrices, we have to "loop" often to do mathematical operations, whereas they are performed in one go using tensors. Any time your code has a loop that is not GPU or TPU friendly.  

Mainstream machine-learning tensor-based frameworks such as PyTorch and TensorFlow, the first thing they want to do, is to convert whatever you're doing into a big matrix multiplication problem, which is where the GPU does its best. Yeah, that's an overstatement, but not so far from reality. Storing all data in tensors and applying batched operations to them is the way to go for hardware acceleration. On GPU, it's a must, and even on CPU, that could give a 10x speedup or more with cache-aware BLAS libraries.

Beyond that, I think there's a lot of basic design improvements that would be necessary before spaGO could fit for mainstream use. Many boilerplates could go away using reflection, or more simply by careful engineering. It's perfectly normal; the more I program in Go, the more I would like to have time to review some choices ...but not that of avoiding tensors, at least for now :)

Why spaGO?
=====

I've been writing more or less the same software for almost 20 years. I guess it's my way of learning a new language. Now it's Go's turn, and spaGO is the result of a few days of pure fun!

Let me explain a little further. It's not precisely the very same software I've been writing now for 20 years: I've been working in the NLP for this long, experimenting with different approaches and techniques, and therefore software of the same field. 
I've always felt satisfied to limit the use of third-party dependencies, trying to implement firsthand the features that interest me most. 
For instance, nowadays, you're nobody if you don't master machine learning. Seriously, NLP's state-of-the-art results have all been achieved with this approach! So, I took the opportunity to speed up my understanding of the underlying deep learning algorithms implementing them almost from scratch in straightforward Go code.
I'm aware that [reinventing the wheel](https://en.wikipedia.org/wiki/Reinventing_the_wheel#Related_phrases) is an anti-pattern; nevertheless, I wanted to build something with my own concepts in my own (italian) style: that's the way I learn best, and it could be your best chance to understand what's going on under the hood of the artificial intelligence :)

When I begin programming in a new language, I know almost anything about it. I often combine the techniques I have acquired by writing in other languages and other paradigms, so some choices may not be the most idiomatic... but who cares, right? 

It's with this approach that I jumped on Go and created spaGo: a work in progress, (hopefully) understandable, easy to use library for machine learning and natural language processing.

Disclaimer
=====

**Please note that I can only do development in my free time** (which is very limited: I am a [#onewheeler](https://twitter.com/hashtag/onewheel), I have a wonderful wife, a [Happy](https://github.com/nlpodyssey/spago/blob/master/assets/happy.jpg) dog, I play the piano and the guitar, and last but not least I'm actively engaged in my [daily job](https://www.exop-group.com/en/)), so no promises are made regarding response time, feature implementations or bug fixes.
If you want spaGo to become something more than just a hobby project of me, I greatly appreciate any bug reports and contributions, which can be made by filing an issue or making a pull request through the github page. Thanks!

Licensing
=====

spaGO is licensed under a BSD-style license. See [LICENSE](https://github.com/nlpodyssey/spago/blob/master/LICENSE) for the full license text.