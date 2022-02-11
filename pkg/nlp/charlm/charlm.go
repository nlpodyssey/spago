// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package charlm provides an implementation of a character-level language model that uses a
// recurrent neural network as its backbone.
// A fully connected softmax layer (a.k.a decoder) is placed on top of each recurrent hidden
// state to predict the next character.
package charlm

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/recurrent/lstm"
	"github.com/nlpodyssey/spago/pkg/nlp/vocabulary"
)

var (
	_ nn.Model[float32] = &Model[float32]{}
)

const (
	// DefaultSequenceSeparator is the default sequence separator value for the
	// character-level language model.
	DefaultSequenceSeparator = "[SEP]"
	// DefaultUnknownToken is the default unknown token value for the
	// character-level language model.
	DefaultUnknownToken = "[UNK]"
)

// Model implements a Character-level Language Model.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Config
	Decoder          *linear.Model[T]
	Projection       *linear.Model[T]
	RNN              *lstm.Model[T]
	Embeddings       []nn.Param[T] `spago:"type:weights;scope:model"`
	Vocabulary       *vocabulary.Vocabulary
	UsedEmbeddings   map[int]ag.Node[T] `spago:"scope:processor"`
	UnknownEmbedding ag.Node[T]         `spago:"scope:processor"`
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new character-level language Model, initialized according to
// the given configuration.
func New[T mat.DType](config Config) *Model[T] {
	if config.SequenceSeparator == "" {
		config.SequenceSeparator = DefaultSequenceSeparator
	}
	if config.UnknownToken == "" {
		config.UnknownToken = DefaultUnknownToken
	}

	if config.OutputSize > 0 {
		// use projection layer
		return &Model[T]{
			Config:     config,
			Decoder:    linear.New[T](config.OutputSize, config.VocabularySize),
			Projection: linear.New[T](config.HiddenSize, config.OutputSize),
			RNN:        lstm.New[T](config.EmbeddingSize, config.HiddenSize),
			Embeddings: newEmptyEmbeddings[T](config.VocabularySize, config.EmbeddingSize),
		}
	}

	// don't use projection layer
	return &Model[T]{
		Config:     config,
		Decoder:    linear.New[T](config.HiddenSize, config.VocabularySize),
		Projection: linear.New[T](config.HiddenSize, config.HiddenSize), // TODO: Find a way to set to nil?
		RNN:        lstm.New[T](config.EmbeddingSize, config.HiddenSize),
		Embeddings: newEmptyEmbeddings[T](config.VocabularySize, config.EmbeddingSize),
	}
}

func newEmptyEmbeddings[T mat.DType](vocabularySize, embeddingSize int) []nn.Param[T] {
	embeddings := make([]nn.Param[T], vocabularySize)
	for i := range embeddings {
		embeddings[i] = nn.NewParam[T](mat.NewEmptyVecDense[T](embeddingSize))
	}
	return embeddings
}

// Initialize initializes the Model m using the given random generator.
func (m *Model[T]) Initialize(rndGen *rand.LockedRand[T]) {
	nn.ForEachParam[T](m, func(param nn.Param[T]) {
		if param.Type() == nn.Weights {
			initializers.XavierUniform(param.Value(), 1, rndGen)
		} else if param.Type() == nn.Biases && param.Name() == "bfor" {
			// LSTM bias hack http://proceedings.mlr.press/v37/jozefowicz15.pdf
			initializers.Constant(param.Value(), 1.0)
		}
	})
}

// InitProcessor initializes embeddings needed by the Forward().
func (m *Model[T]) InitProcessor() {
	m.UsedEmbeddings = make(map[int]ag.Node[T])
	m.UnknownEmbedding = m.Graph().NewWrap(m.Embeddings[m.Vocabulary.MustID(m.UnknownToken)])
}

// Forward performs the forward step for each input and returns the result.
func (m *Model[T]) Forward(in interface{}) interface{} {
	xs := in.([]string)
	ys := make([]ag.Node[T], len(xs))
	encoding := m.GetEmbeddings(xs)
	for i, x := range encoding {
		m.Graph().IncTimeStep() // essential for truncated back-propagation
		h := m.RNN.Forward(x)
		proj := nn.ToNode[T](m.UseProjection(h))
		ys[i] = nn.ToNode[T](m.Decoder.Forward(proj))
	}
	return ys
}

// UseProjection performs a linear projection with Processor.Projection model,
// if available, otherwise returns xs unmodified.
func (m *Model[T]) UseProjection(xs []ag.Node[T]) []ag.Node[T] {
	if m.Config.OutputSize > 0 {
		return m.Projection.Forward(xs...)
	}
	return xs
}

// GetEmbeddings transforms the string sequence xs into a sequence of
// embeddings nodes.
func (m *Model[T]) GetEmbeddings(xs []string) []ag.Node[T] {
	ys := make([]ag.Node[T], len(xs))
	for i, item := range xs {
		id, ok := m.Vocabulary.ID(item)
		if !ok {
			ys[i] = m.UnknownEmbedding
			continue
		}
		if embedding, ok := m.UsedEmbeddings[id]; ok {
			ys[i] = embedding
			continue
		}
		ys[i] = m.Graph().NewWrap(m.Embeddings[id])
		m.UsedEmbeddings[id] = ys[i]
	}
	return ys
}
