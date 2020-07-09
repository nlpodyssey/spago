# Documentation Index

* [Godocs for spaGo](https://pkg.go.dev/mod/github.com/nlpodyssey/spago)
* [Contributing to spaGo](../CONTRIBUTING.md)

## Demos

Every demo can be built and run separately. Alternatively, all of the demos can be built and packaged together as a single Docker image. When using the Docker image, the demos can be invoked separately.

* [Named Entities Recognition](named-entities-recognition-demo.md)
* [Import a Pre-Trained Model](import-a-pre-trained-model-demo.md)
* [Question Answering](question-answering-demo.md)
* [Masked Language Model](masked-language-model-demo.md)

## Feature Source Tree

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
