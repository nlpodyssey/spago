// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// CharLM implements a character-level language model that uses a recurrent neural network as its backbone.
// A fully connected softmax layer (a.k.a decoder) is placed on top of each recurrent hidden state to predict
// the next character.
package charlm

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/rec/lstm"
	"github.com/nlpodyssey/spago/pkg/nlp/vocabulary"
)

const (
	DefaultSequenceSeparator = "[SEP]"
	DefaultUnknownToken      = "[UNK]"
)

type Model struct {
	Config
	Decoder    *linear.Model
	RNN        nn.Model
	Embeddings []*nn.Param `type:"weights"`
	Vocabulary *vocabulary.Vocabulary
}

// TODO: add dropout
type Config struct {
	VocabularySize    int
	EmbeddingSize     int
	HiddenSize        int
	OutputSize        int    // TODO: is it always equal to the vocabulary size?
	SequenceSeparator string // empty string is replaced with DefaultSequenceSeparator
	UnknownToken      string // empty string is replaced with DefaultUnknownToken
}

func New(config Config) *Model {
	if config.SequenceSeparator == "" {
		config.SequenceSeparator = DefaultSequenceSeparator
	}
	if config.UnknownToken == "" {
		config.UnknownToken = DefaultUnknownToken
	}
	return &Model{
		Config:     config,
		Decoder:    linear.New(config.HiddenSize, config.OutputSize),
		RNN:        lstm.New(config.EmbeddingSize, config.HiddenSize),
		Embeddings: newEmptyEmbeddings(config.VocabularySize, config.EmbeddingSize),
	}
}

func newEmptyEmbeddings(vocabularySize, embeddingSize int) []*nn.Param {
	embeddings := make([]*nn.Param, vocabularySize)
	for i := range embeddings {
		embeddings[i] = nn.NewParam(mat.NewEmptyVecDense(embeddingSize))
	}
	return embeddings
}

func Initialize(m *Model, rndGen *rand.LockedRand) {
	nn.ForEachParam(m, func(param *nn.Param) {
		if param.Type() == nn.Weights {
			initializers.XavierUniform(param.Value(), 1, rndGen)
		} else if param.Type() == nn.Biases && param.Name() == "bfor" {
			// LSTM bias hack http://proceedings.mlr.press/v37/jozefowicz15.pdf
			initializers.Constant(param.Value(), 1.0)
		}
	})
}

type Processor struct {
	nn.BaseProcessor
	Decoder          *linear.Processor
	RNN              nn.Processor
	usedEmbeddings   map[int]ag.Node
	UnknownEmbedding ag.Node
}

func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	p := &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: false,
		},
		Decoder:          m.Decoder.NewProc(g).(*linear.Processor),
		RNN:              m.RNN.NewProc(g),
		usedEmbeddings:   make(map[int]ag.Node),
		UnknownEmbedding: g.NewWrap(m.Embeddings[m.Vocabulary.MustId(m.UnknownToken)]),
	}
	return p
}

func (p *Processor) SetMode(mode nn.ProcessingMode) {
	p.Mode = mode
	nn.SetProcessingMode(mode, p.RNN, p.Decoder)
}

func (p *Processor) Predict(xs ...string) []ag.Node {
	ys := make([]ag.Node, len(xs))
	encoding := p.GetEmbeddings(xs)
	for i, x := range encoding {
		p.Graph.IncTimeStep() // essential for truncated back-propagation
		h := p.RNN.Forward(x)[0]
		ys[i] = p.Decoder.Forward(h)[0]
	}
	return ys
}

func (p *Processor) GetEmbeddings(xs []string) []ag.Node {
	model := p.Model.(*Model)
	ys := make([]ag.Node, len(xs))
	for i, item := range xs {
		id, ok := model.Vocabulary.Id(item)
		if !ok {
			ys[i] = p.UnknownEmbedding
			continue
		}
		if embedding, ok := p.usedEmbeddings[id]; ok {
			ys[i] = embedding
			continue
		}
		ys[i] = p.Graph.NewWrap(model.Embeddings[id])
		p.usedEmbeddings[id] = ys[i]
	}
	return ys
}

func (p *Processor) Forward(_ ...ag.Node) []ag.Node {
	panic("charlm: method not implemented. Use Predict() instead.")
}
