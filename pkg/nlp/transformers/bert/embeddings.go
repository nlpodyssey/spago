// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
)

var (
	_ nn.Model = &Embeddings{}
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
type Embeddings struct {
	nn.BaseModel
	EmbeddingsConfig
	Words            *embeddings.Model
	Position         []nn.Param `spago:"type:weights"` // TODO: stop auto-wrapping
	TokenType        []nn.Param `spago:"type:weights"`
	Norm             *layernorm.Model
	Projector        *linear.Model
	UnknownEmbedding ag.Node `spago:"scope:processor"`
}

func init() {
	gob.Register(&Embeddings{})
}

// NewEmbeddings returns a new BERT Embeddings model.
func NewEmbeddings(config EmbeddingsConfig) *Embeddings {
	return &Embeddings{
		EmbeddingsConfig: config,
		Words: embeddings.New(embeddings.Config{
			Size:       config.Size,
			DBPath:     config.WordsMapFilename,
			ReadOnly:   config.WordsMapReadOnly,
			ForceNewDB: config.DeletePreEmbeddings,
		}),
		Position:  newPositionEmbeddings(config.Size, config.MaxPositions),
		TokenType: newTokenTypes(config.Size, config.TokenTypes),
		Norm:      layernorm.New(config.Size),
		Projector: newProjector(config.Size, config.OutputSize),
	}
}

// InitProcessor initializes the unknown embeddings.
func (m *Embeddings) InitProcessor() {
	m.UnknownEmbedding = m.Graph().NewWrap(m.Words.GetStoredEmbedding(wordpiecetokenizer.DefaultUnknownToken))
}

func newPositionEmbeddings(size, maxPositions int) []nn.Param {
	out := make([]nn.Param, maxPositions)
	for i := 0; i < maxPositions; i++ {
		out[i] = nn.NewParam(mat.NewEmptyVecDense(size))
	}
	return out
}

func newTokenTypes(size, tokenTypes int) []nn.Param {
	out := make([]nn.Param, tokenTypes)
	for i := 0; i < tokenTypes; i++ {
		out[i] = nn.NewParam(mat.NewEmptyVecDense(size))
	}
	return out
}

func newProjector(in, out int) *linear.Model {
	if in == out {
		return nil // projection not needed
	}
	return linear.New(in, out)
}

// Encode transforms a string sequence into an encoded representation.
func (m *Embeddings) Encode(words []string) []ag.Node {
	encoded := make([]ag.Node, len(words))
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

func (m *Embeddings) getWordEmbeddings(words []string) []ag.Node {
	out := make([]ag.Node, len(words))
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

func (m *Embeddings) useProjection(xs []ag.Node) []ag.Node {
	if m.Projector == nil {
		return xs
	}
	return m.Projector.Forward(xs...)
}
