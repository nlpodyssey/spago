spaGO
=====
![Go](https://github.com/nlpodyssey/spago/workflows/Go/badge.svg?branch=master)
[![Go Report Card](https://goreportcard.com/badge/github.com/nlpodyssey/spago)](https://goreportcard.com/report/github.com/nlpodyssey/spago)
[![Maintainability](https://api.codeclimate.com/v1/badges/be7350d3eb1a6a8aa503/maintainability)](https://codeclimate.com/github/nlpodyssey/spago/maintainability)
[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)
![Unstable](https://github.com/nlpodyssey/spago/blob/main/assets/stability-unstable-yellow.svg)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

<p align="center"><i>If you like the project, please ★ star this repository to show your support! 🤩</i></p>

A beautiful and maintainable machine learning library written in Go. It is designed to support relevant neural architectures in **Natural Language Processing**.

spaGO is compatible with 🤗 BERT-like [Transformers](https://github.com/huggingface/transformers) and with the [Flair](https://github.com/flairNLP/flair) sequence labeler architecture. 

<div style="text-align:center"><img src="https://github.com/nlpodyssey/spago/blob/main/assets/screenshot_spago_api_qa.png" /></div>

---

spaGO ships with a ton of built-in features, including:
* **Automatic differentiation**. You write the *forward()*, it does all *backward()* derivatives for you:
    -   Define-by-Run (default, just like PyTorch does)
    -   Define-and-Run (similar to the static graph of TensorFlow)
* **Optimization methods**:
    - Adam, RAdam, RMS-Prop, AdaGrad, SGD
* **Neural networks**:
    -   Feed-forward models (Linear, Highway, Convolution, ...)
    -   Recurrent models (LSTM, GRU, BiLSTM...)
    -   Attention mechanisms (Self-Attention, Multi-Head Attention, ...)
    -   Recursive autoencoder
* **Natural Language Processing**:
    -   Memory-efficient Word Embeddings (with [badger](https://github.com/dgraph-io/badger) key–value store)
    -   Character Language Models
    -   Recurrent Sequence Labeler with CRF on top (e.g. Named Entities Recognition)
    -   Transformer models (BERT-like)
        -   Masked language model
        -   Next sentence prediction
        -   Tokens Classification
        -   Question Answering

...and the best is yet to come!

Installation
=====
Make sure you have [Go 1.14](https://golang.org/dl/) installed on your computer first. The package can be installed using *go get* as follows:

```console
go get -u github.com/nlpodyssey/spago
```

spaGO is compatible with [go modules](https://blog.golang.org/using-go-modules).

Demo for Named Entities Recognition
=====

To evaluate the usability of spaGO in NLP, I began experimenting with a basic task such as sequence labeling applied to [Named Entities Recognition (NER)](https://en.wikipedia.org/wiki/Named-entity_recognition).

I felt the need to achieve gratification as quickly as possible, so I opted to use the state-of-the-art pre-trained model released with the [Flair](https://github.com/flairNLP/flair) library, instead of training one from scratch.

You got it, I wrote a program to import the parameters (weights and bias) of Flair into spaGO structures. I'll make it available soon, now it's a bit chaotic.    

Before start, make sure you have Go 1.14 and spaGO installed (or just cloned).

### Build

Move into the spaGO directory.

If you're on Linux and AMD64 architecture run:

```consoleyou got it
GOOS=linux GOARCH=amd64 go build -o ner-server cmd/ner/main.go 
```

If the command is successful you should find an executable called `ner-server` in the same folder.

You can change the `GOOS` and `GOARCH` according to the [build](https://golang.org/pkg/go/build/) documentation but please note that I have so far tested only with Linux on AMD64.

### Docker Build

Alternativelty, the demos can be built into a single Docker container image.

Move into the spaGO directory and run:

```console
docker build -t spago:main . -f Dockerfile
```

### Run

You must indicate the directory that contains the spaGO neural models. Reasonably, you don't have this folder yet, so you can create a new one, for example:

```console
mkdir ~/.spago 
```

Now run the `ner-server` indicating a port, the directory of the models, and the model name.

At present, there are two models available, named `goflair-en-ner-conll03` and `goflair-en-ner-fast-conll03`.

Example: 
 
```console
./ner-server 1987 ~/.spago goflair-en-ner-fast-conll03
```

It should print:

```console
Fetch model from `https://dl.dropboxusercontent.com/s/9lhh9uom6vh66pg/goflair-en-ner-fast-conll03.tar.gz?dl=0`
Downloading... 278 MB complete     
Extracting compressed model... ok
Loading model parameters from `~/.spago/goflair-en-ner-fast-conll03/model.bin`... ok
Start server on port 1987.
```

At the first execution, the program downloads the required model, if available. For successive executions, it uses the previously downloaded model.

### Docker Run

You must indicate the directory that contains the spaGO neural models. Reasonably, you don't have this folder yet, so you can create a new one, for example:

```console
mkdir ~/.spago 
```

Now run the `ner-server` using the Docker container image, indicating a port, the directory of the models, and the model name. Include the volume flag (`-v`) to bind-mount the directory of the models into the container, and include the publish flag (`-p`) to publish the server port to the container's host.

```console
docker run --rm -it -p:1987:1987 -v ~/.spago:/tmp/spago spago:main ./ner-server 1987 /tmp/spago goflair-en-ner-fast-conll03
```

### API

You can test the API from command line with curl:

```console
curl -d '{"options": {"mergeEntities": true, "filterNotEntities": true}, "text": "Mark Freuder Knopfler was born in Glasgow, Scotland, to an English mother, Louisa Mary, and a Jewish Hungarian father, Erwin Knopfler. He was the lead guitarist, singer, and songwriter for the rock band Dire Straits"}' -H "Content-Type: application/json" "http://127.0.0.1:1987/analyze?pretty"
```

It should print:

```json
{
    "tokens": [
        {
            "text": "Mark Freuder Knopfler",
            "start": 0,
            "end": 21,
            "label": "PER"
        },
        {
            "text": "Glasgow",
            "start": 34,
            "end": 41,
            "label": "LOC"
        },
        {
            "text": "Scotland",
            "start": 43,
            "end": 51,
            "label": "LOC"
        },
        {
            "text": "English",
            "start": 59,
            "end": 66,
            "label": "MISC"
        },
        {
            "text": "Louisa Mary",
            "start": 75,
            "end": 86,
            "label": "PER"
        },
        {
            "text": "Jewish",
            "start": 94,
            "end": 100,
            "label": "MISC"
        },
        {
            "text": "Hungarian",
            "start": 101,
            "end": 110,
            "label": "MISC"
        },
        {
            "text": "Erwin Knopfler",
            "start": 119,
            "end": 133,
            "label": "PER"
        },
        {
            "text": "Dire Straits",
            "start": 203,
            "end": 215,
            "label": "ORG"
        }
    ]
}
```

Demo for Question Answering
=====

Until recently, question-answering was considered a complex task. Today you can get good results with just a [linear layer](https://github.com/nlpodyssey/spago/blob/main/pkg/nlp/transformers/bert/spanclassifier.go#L25) on top of the transformer's encoding. Transformers are a recent trend in natural language processing. They are auto-regressive models trained in an unsupervised manner on huge amounts of text to assimilate human language patterns. In other words, they are [super-parrots](https://medium.com/@ElementalCognition/can-super-parrots-ever-achieve-language-understanding-8307dfd3e87c). Although I do not believe that this is the right way to solve the problem of language processing - at least not alone - I have to admit that their power is extraordinary. 

No more talk. Here's how to test a question-answering system based on BERT, the first Transformer. 

Before start, make sure you have Go 1.14 and spaGO installed (or just cloned).

### Build

Change into the spaGO directory.

If you're on Linux and AMD64 architecture run:

```console
GOOS=linux GOARCH=amd64 go build -o hugging_face_importer cmd/huggingfaceimporter/main.go
```

```console
GOOS=linux GOARCH=amd64 go build -o bert_server cmd/bert/main.go
```

If the commands are successful you should find two executables called `hugging_face_importer` and `bert_server` in the same folder.

You can change the `GOOS` and `GOARCH` according to the [build](https://golang.org/pkg/go/build/) documentation but please note that I have so far tested only with Linux on AMD64.

### Docker Build

Alternativelty, the demos can be built into a single Docker container image.

If the Docker container image hasn't already been built, move into the spaGO directory and run:

```console
docker build -t spago:main . -f Dockerfile
```

### Import a Pre-Trained Model

spaGO allows you either to use a model in the inference phase or to train one from scratch, or fine-tune it.
However, training a language model (i.e. the transformer objective) to get competitive results can become prohibitive.
This applies in general, but even more so with spaGO as it does not currently use the GPU :scream:

Pre-trained transformer models fine-tuned for question-answering exist for several languages and are publicly hosted on the [Hugging Face models repository](https://huggingface.co/models). Particularly, these exist for BERT and ELECTRA, the two types of transformers currently supported by spaGO.

To import a pre-trained model, run the `hugging_face_importer` indicating both the model name you'd like to import (including organization), and a local directory where to store all your models.

Example: 

```console
./hugging_face_importer --model=deepset/bert-base-cased-squad2 --repo=~/.spago 
```

At the end of the process, you should see:

```console
Serializing model to "~/.spago/deepset/bert-base-cased-squad2/spago_model.bin"... ok
Cool! 🤗 transformer has been successfully converted!
```

### Import a Pre-Trained Model using Docker

Run the container image, including the volume flag (`-v`) to bind-mount the directory of the models into the container.

Example:

```console
docker run --rm -it -v ~/.spago:/tmp/spago spago:main ./hugging_face_importer --model=deepset/bert-base-cased-squad2 --repo=/tmp/spago
```

### Run

If you followed the import step above, now you should see the directory `~/.spago/deepset/bert-base-cased-squad2` containing the original Hugging Face files plus the files generated by spaGO: `spago_model.bin` and `embeddings_storage`. 

Run the `bert_server` indicating a port and the model path (NOT the model file).

Example: 
 
```console
./bert_server --model=~/.spago/deepset/bert-base-cased-squad2 --tls-disable
```

It should print:

```console
TLS Cert path is /etc/ssl/certs/spago/server.crt
TLS private key path is /etc/ssl/certs/spago/server.key
Start loading pre-trained model from "~/.spago/deepset/bert-base-cased-squad2"
[1/3] Loading configuration... ok
[2/3] Loading vocabulary... ok
[3/3] Loading model weights... ok
Config: {HiddenAct:gelu HiddenSize:768 IntermediateSize:3072 MaxPositionEmbeddings:512 NumAttentionHeads:12 NumHiddenLayers:12 TypeVocabSize:2 VocabSize:28996}
Start non-TLS server listening on 0.0.0.0:1987.
```

### Docker Run

Run the container image, including the volume flag (`-v`) to bind-mount the directory of the models into the container, and including the publish flag (`-p`) to publish the server port to the container's host. TLS is enabled by default.

Example:

```console
docker run --rm -it -p 1987:1987 -v ~/.spago:/tmp/spago spago:main ./bert_server --model=/tmp/spago/deepset/bert-base-cased-squad2
```

### API

You can easily test the API with the command line using curl.

Set a PASSAGE and a couple of QUESTIONS as environment variables:

```console
PASSAGE="BERT is a technique for NLP developed by Google. BERT was created and published in 2018 by Jacob Devlin and his colleagues from Google."
QUESTION1="Who is the author of BERT?"
QUESTION2="When was BERT created?"
```

To get the answer to the first question, execute:

```console
curl -k -d '{"question": "'"$QUESTION1"'", "passage": "'"$PASSAGE"'"}' -H "Content-Type: application/json" "https://127.0.0.1:1987/answer?pretty"
```

It should print:

```json
{
    "answers": [
        {
            "text": "Jacob Devlin",
            "start": 91,
            "end": 103,
            "confidence": 0.9641588621246571
        }
    ]
}
```

To get the answer to the second question, execute:

```console
curl -k -d '{"question": "'"$QUESTION2"'", "passage": "'"$PASSAGE"'"}' -H "Content-Type: application/json" "https://127.0.0.1:1987/answer?pretty"
```

It should print:

```json
{
    "answers": [
        {
            "text": "2018",
            "start": 83,
            "end": 87,
            "confidence": 0.9924210921706913
        }
    ]
}
```

At the heart of the library
=====

The [ag package](https://github.com/nlpodyssey/spago/tree/main/pkg/ml/ag) (a.k.a. auto-grad) is the centerpiece of the spaGO machine learning framework.

Neural models optimized by back-propagation require gradients to be available during training.
The set of expressions characterizing the forward-step of such models must be defined within the [ag.Graph](https://github.com/nlpodyssey/spago/blob/main/pkg/ml/ag/graph.go) to take advantage of automatic differentiation.

### The basis

Let's see if spaGO can tell us what two plus five is :)

```go
// create a new node of type variable with a scalar
a := ag.NewVariable(mat.NewScalar(2.0), true)
// create another node of type variable with a scalar
b := ag.NewVariable(mat.NewScalar(5.0), true)
// create an addition operator (the calculation is actually performed here)
c := ag.Add(a, b)
// print the result
fmt.Printf("c = %v\n", c.Value())
```

It should print:

```console
c = [7]
```

Let's go one step further now and ask spaGO to give us the gradients on `a` and `b`, starting with arbitrary output gradients.

```go
ag.Backward(c, ag.OutputGrad(mat.NewScalar(0.5)))
fmt.Printf("ga = %v\n", a.Grad())
fmt.Printf("gb = %v\n", b.Grad())
```

It should print:

```console
ga = [0.5]
gb = [0.5]
```

I know it's a tiny example, but have patience. You will soon find some tutorials on the [Wiki](https://github.com/nlpodyssey/spago/wiki/Machine-Learning-Framework).

What's inside?
=====

I haven't found the time yet to write a proper documentation, or at least a clear description of what spaGO contains. I recently started a timid attempt to write a few lines on the [Wiki](https://github.com/nlpodyssey/spago/wiki/Machine-Learning-Framework).
 
I'm trying to keep the code self-documenting and straightforward through. By that, I don't mean that I don't have to improve the documentation tremendously. It's my very next step.

For the time being, I hope that a tree-like view of the library (*pkg* folder) can help you to understand the current supported features, and - more important - how I decided to structure spaGO in the first place.

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

Current status
=====
We're not at a v1.0.0 yet, so spaGO is currently an experimental work-in-progress. 
It's pretty easy to get your hands on through, so you might want to use it in your real applications. Early adopters may make use of it for production use today as long as they understand and accept that spaGO is not fully tested and that APIs will change (maybe extensively).

If you're wondering, I haven't used spaGO in production yet, but I plan to do the first integration tests soon.

Blah, blah, blah
=====

### Why spaGO?

I've been writing more or less the same software for almost 20 years. I guess it's my way of learning a new language. Now it's Go's turn, and spaGO is the result of a few days of pure fun!

Let me explain a little further. It's not precisely the very same software I've been writing now for 20 years: I've been working in the NLP for this long, experimenting with different approaches and techniques, and therefore software of the same field. 
I've always felt satisfied to limit the use of third-party dependencies, writing firsthand the algorithms that interest me most. 
So, I took the opportunity to speed up my understanding of the deep learning techniques and methodologies underlying cutting-edge NLP results, implementing them almost from scratch in straightforward Go code.
I'm aware that [reinventing the wheel](https://en.wikipedia.org/wiki/Reinventing_the_wheel#Related_phrases) is an anti-pattern; nevertheless, I wanted to build something with my own concepts in my own (italian) style: that's the way I learn best, and it could be your best chance to understand what's going on under the hood of the artificial intelligence :)

When I start programming in a new language, I usually do not know much of it. I often combine the techniques I have acquired by writing in other languages and other paradigms, so some choices may not be the most idiomatic... but who cares, right? 

It's with this approach that I jumped on Go and created spaGo: a work in progress, (hopefully) understandable, easy to use library for machine learning and natural language processing.

### Is spaGO right for me?

Are you looking for a highly optimized, scalable, battle-tested, production-ready machine-learning/NLP framework? Are you also a Python lover and enjoy manipulating tensors? If yes, you won't find much to your satisfaction here. [PyTorch](https://pytorch.org/) plus the wonders of the friends of [Hugging Face](https://github.com/huggingface) is the answer you seek!

If instead you prefer statically typed, compiled programming language, and a **simpler yet well-structured** machine-learning framework almost ready to use is what you need, then you are in the right place!

The idea is that you could have written spaGO. Most of it, from the computational graph to the [LSTM](https://github.com/nlpodyssey/spago/blob/main/pkg/ml/nn/rec/lstm/lstm.go#L182) is straightforward Go code :)

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

Disclaimer
=====

**Please note that I can only do development in my free time** (which is very limited: I am a [#onewheeler](https://twitter.com/hashtag/onewheel), I have a wonderful wife, a [Happy](https://github.com/nlpodyssey/spago/blob/main/assets/happy.jpg) dog, I play the piano and the guitar, and last but not least I'm actively engaged in my [daily job](https://www.exop-group.com/en/)), so no promises are made regarding response time, feature implementations or bug fixes.
If you want spaGo to become something more than just a hobby project of me, I greatly appreciate any bug reports and contributions, which can be made by filing an issue or making a pull request through the github page. Thanks!

Contact
===== 

I encourage you to write an issue. This would help the community grow.

If you really want to write to me privately, please email [Matteo Grella](mailto:matteogrella@gmail.com) with your questions or comments.

Licensing
=====

spaGO is licensed under a BSD-style license. See [LICENSE](https://github.com/nlpodyssey/spago/blob/main/LICENSE) for the full license text.
