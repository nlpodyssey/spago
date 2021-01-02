// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package charlm provides an implementation of a character-level language model that uses a
// recurrent neural network as its backbone.
// A fully connected softmax layer (a.k.a decoder) is placed on top of each recurrent hidden
// state to predict the next character.
package charlm

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/recurrent/lstm"
	"github.com/nlpodyssey/spago/pkg/nlp/vocabulary"
)

const (
	// DefaultSequenceSeparator is the default sequence separator value for the
	// character-level language model.
	DefaultSequenceSeparator = "[SEP]"
	// DefaultUnknownToken is the default unknown token value for the
	// character-level language model.
	DefaultUnknownToken = "[UNK]"
)

// Config provides configuration settings for a Character-level Language Model.
// TODO: add dropout
type Config struct {
	VocabularySize    int
	EmbeddingSize     int
	HiddenSize        int
	OutputSize        int    // use the projection layer when the output size is > 0
	SequenceSeparator string // empty string is replaced with DefaultSequenceSeparator
	UnknownToken      string // empty string is replaced with DefaultUnknownToken
}

// Model implements a Character-level Language Model.
type Model struct {
	nn.BaseModel
	Config
	Decoder          *linear.Model
	Projection       *linear.Model
	RNN              *lstm.Model
	Embeddings       []nn.Param `spago:"type:weights;scope:model"`
	Vocabulary       *vocabulary.Vocabulary
	UsedEmbeddings   map[int]ag.Node `spago:"scope:processor"`
	UnknownEmbedding ag.Node         `spago:"scope:processor"`
}

// New returns a new character-level language Model, initialized according to
// the given configuration.
func New(config Config) *Model {
	if config.SequenceSeparator == "" {
		config.SequenceSeparator = DefaultSequenceSeparator
	}
	if config.UnknownToken == "" {
		config.UnknownToken = DefaultUnknownToken
	}

	if config.OutputSize > 0 {
		// use projection layer
		return &Model{
			Config:     config,
			Decoder:    linear.New(config.OutputSize, config.VocabularySize),
			Projection: linear.New(config.HiddenSize, config.OutputSize),
			RNN:        lstm.New(config.EmbeddingSize, config.HiddenSize),
			Embeddings: newEmptyEmbeddings(config.VocabularySize, config.EmbeddingSize),
		}
	}

	// don't use projection layer
	return &Model{
		Config:     config,
		Decoder:    linear.New(config.HiddenSize, config.VocabularySize),
		Projection: linear.New(config.HiddenSize, config.HiddenSize), // TODO: Find a way to set to nil?
		RNN:        lstm.New(config.EmbeddingSize, config.HiddenSize),
		Embeddings: newEmptyEmbeddings(config.VocabularySize, config.EmbeddingSize),
	}
}

func newEmptyEmbeddings(vocabularySize, embeddingSize int) []nn.Param {
	embeddings := make([]nn.Param, vocabularySize)
	for i := range embeddings {
		embeddings[i] = nn.NewParam(mat.NewEmptyVecDense(embeddingSize))
	}
	return embeddings
}

// Initialize initializes the Model m using the given random generator.
func Initialize(m *Model, rndGen *rand.LockedRand) {
	nn.ForEachParam(m, func(param nn.Param) {
		if param.Type() == nn.Weights {
			initializers.XavierUniform(param.Value(), 1, rndGen)
		} else if param.Type() == nn.Biases && param.Name() == "bfor" {
			// LSTM bias hack http://proceedings.mlr.press/v37/jozefowicz15.pdf
			initializers.Constant(param.Value(), 1.0)
		}
	})
}

// InitProcessor initializes embeddings needed by the Forward().
func (m *Model) InitProcessor() {
	m.UsedEmbeddings = make(map[int]ag.Node)
	m.UnknownEmbedding = m.Graph().NewWrap(m.Embeddings[m.Vocabulary.MustID(m.UnknownToken)])
}

// Forward performs the forward step for each input and returns the result.
func (m *Model) Forward(in interface{}) interface{} {
	xs := in.([]string)
	ys := make([]ag.Node, len(xs))
	encoding := m.GetEmbeddings(xs)
	for i, x := range encoding {
		m.Graph().IncTimeStep() // essential for truncated back-propagation
		h := m.RNN.Forward(x)
		proj := nn.ToNode(m.UseProjection(h))
		ys[i] = nn.ToNode(m.Decoder.Forward(proj))
	}
	return ys
}

// UseProjection performs a linear projection with Processor.Projection model,
// if available, otherwise returns xs unmodified.
func (m *Model) UseProjection(xs []ag.Node) []ag.Node {
	if m.Config.OutputSize > 0 {
		return m.Projection.Forward(xs...)
	}
	return xs
}

// GetEmbeddings transforms the string sequence xs into a sequence of
// embeddings nodes.
func (m *Model) GetEmbeddings(xs []string) []ag.Node {
	ys := make([]ag.Node, len(xs))
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
