// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package learnedpositionalencoder

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model[float32] = &LearnedPositionalEncoder[float32]{}
)

// Config provides configuration settings for a LearnedPositionalEncoder Model.
type Config struct {
	NumEmbeddings int
	EmbeddingDim  int
	PaddingIDX    int
	Offset        int
}

// LearnedPositionalEncoder contains positional embeddings fine-tuned during
// the training phase.
type LearnedPositionalEncoder[T mat.DType] struct {
	nn.BaseModel[T]
	Config  Config
	Vectors []nn.Param[T] `spago:"type:weights;scope:model"`
}

func init() {
	gob.Register(&LearnedPositionalEncoder[float32]{})
	gob.Register(&LearnedPositionalEncoder[float64]{})
}

// New returns a new LearnedPositionalEncoder.
// TODO: PaddingIDX
func New[T mat.DType](config Config) *LearnedPositionalEncoder[T] {
	vectors := make([]nn.Param[T], config.NumEmbeddings+config.Offset)
	for i := 0; i < len(vectors); i++ {
		vectors[i] = nn.NewParam[T](mat.NewEmptyVecDense[T](config.EmbeddingDim))
	}
	return &LearnedPositionalEncoder[T]{
		Config:  config,
		Vectors: vectors,
	}
}

// Encode performs the forward step for each input and returns the result.
func (m *LearnedPositionalEncoder[T]) Encode(positions []int) []ag.Node[T] {
	g := m.Graph()
	embeddings := make([]ag.Node[T], len(positions))
	for i, pos := range positions {
		embeddings[i] = g.NewWrap(m.Vectors[pos+m.Config.Offset])
	}
	return embeddings
}
