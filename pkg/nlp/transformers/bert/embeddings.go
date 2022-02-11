// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
)

var (
	_ nn.Model[float32] = &Embeddings[float32]{}
)

// EmbeddingsConfig provides configuration settings for BERT Embeddings.
type EmbeddingsConfig struct {
	Size                int
	OutputSize          int
	MaxPositions        int
	TokenTypes          int
	WordsMapFilename    string
	WordsMapReadOnly    bool
	DeletePreEmbeddings bool
}

// Embeddings is a BERT Embeddings model.
type Embeddings[T mat.DType] struct {
	nn.BaseModel[T]
	EmbeddingsConfig
	Words            *embeddings.Model[T]
	Position         []nn.Param[T] `spago:"type:weights"` // TODO: stop auto-wrapping
	TokenType        []nn.Param[T] `spago:"type:weights"`
	Norm             *layernorm.Model[T]
	Projector        *linear.Model[T]
	UnknownEmbedding ag.Node[T] `spago:"scope:processor"`
}

func init() {
	gob.Register(&Embeddings[float32]{})
	gob.Register(&Embeddings[float64]{})
}

// NewEmbeddings returns a new BERT Embeddings model.
func NewEmbeddings[T mat.DType](config EmbeddingsConfig) *Embeddings[T] {
	return &Embeddings[T]{
		EmbeddingsConfig: config,
		Words: embeddings.New[T](embeddings.Config{
			Size:       config.Size,
			DBPath:     config.WordsMapFilename,
			ReadOnly:   config.WordsMapReadOnly,
			ForceNewDB: config.DeletePreEmbeddings,
		}),
		Position:  newPositionEmbeddings[T](config.Size, config.MaxPositions),
		TokenType: newTokenTypes[T](config.Size, config.TokenTypes),
		Norm:      layernorm.New[T](config.Size),
		Projector: newProjector[T](config.Size, config.OutputSize),
	}
}

// InitProcessor initializes the unknown embeddings.
func (m *Embeddings[_]) InitProcessor() {
	m.UnknownEmbedding = m.Graph().NewWrap(m.Words.GetStoredEmbedding(wordpiecetokenizer.DefaultUnknownToken))
}

func newPositionEmbeddings[T mat.DType](size, maxPositions int) []nn.Param[T] {
	out := make([]nn.Param[T], maxPositions)
	for i := 0; i < maxPositions; i++ {
		out[i] = nn.NewParam[T](mat.NewEmptyVecDense[T](size))
	}
	return out
}

func newTokenTypes[T mat.DType](size, tokenTypes int) []nn.Param[T] {
	out := make([]nn.Param[T], tokenTypes)
	for i := 0; i < tokenTypes; i++ {
		out[i] = nn.NewParam[T](mat.NewEmptyVecDense[T](size))
	}
	return out
}

func newProjector[T mat.DType](in, out int) *linear.Model[T] {
	if in == out {
		return nil // projection not needed
	}
	return linear.New[T](in, out)
}

// Encode transforms a string sequence into an encoded representation.
func (m *Embeddings[T]) Encode(words []string) []ag.Node[T] {
	encoded := make([]ag.Node[T], len(words))
	wordEmbeddings := m.getWordEmbeddings(words)
	sequenceIndex := 0
	for i := 0; i < len(words); i++ {
		encoded[i] = wordEmbeddings[i]
		encoded[i] = m.Graph().Add(encoded[i], m.Graph().NewWrap(m.Position[i]))
		encoded[i] = m.Graph().Add(encoded[i], m.TokenType[sequenceIndex])
		if words[i] == wordpiecetokenizer.DefaultSequenceSeparator {
			sequenceIndex++
		}
	}
	return m.useProjection(m.Norm.Forward(encoded...))
}

func (m *Embeddings[T]) getWordEmbeddings(words []string) []ag.Node[T] {
	out := make([]ag.Node[T], len(words))
	for i, embedding := range m.Words.Encode(words) {
		switch embedding {
		case nil:
			out[i] = m.UnknownEmbedding
		default:
			out[i] = embedding
		}
	}
	return out
}

func (m *Embeddings[T]) useProjection(xs []ag.Node[T]) []ag.Node[T] {
	if m.Projector == nil {
		return xs
	}
	return m.Projector.Forward(xs...)
}
