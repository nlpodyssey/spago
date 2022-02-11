// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sinusoidalpositionalencoder

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/encoding/pe"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model[float32] = &SinusoidalPositionalEncoder[float32]{}
)

// Config provides configuration settings for a SinusoidalPositionalEncoder Model.
type Config struct {
	NumEmbeddings int
	EmbeddingDim  int
}

// SinusoidalPositionalEncoder contains positional embeddings fine-tuned during
// the training phase.
type SinusoidalPositionalEncoder[T mat.DType] struct {
	nn.BaseModel[T]
	Config   Config
	Delegate *pe.SinusoidalPositionalEncoder[T]
}

func init() {
	gob.Register(&SinusoidalPositionalEncoder[float32]{})
	gob.Register(&SinusoidalPositionalEncoder[float64]{})
}

// New returns a new SinusoidalPositionalEncoder.
func New[T mat.DType](config Config) *SinusoidalPositionalEncoder[T] {
	return &SinusoidalPositionalEncoder[T]{
		Config:   config,
		Delegate: pe.NewSinusoidalPositionalEncoder[T](config.EmbeddingDim, config.NumEmbeddings),
	}
}

// Encode performs the forward step for each input and returns the result.
func (m *SinusoidalPositionalEncoder[T]) Encode(positions []int) []ag.Node[T] {
	embeddings := make([]ag.Node[T], len(positions))
	for i, vector := range m.Delegate.Encode(positions...) {
		embeddings[i] = m.Graph().NewVariable(vector, false)
	}
	return embeddings
}
