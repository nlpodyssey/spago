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
	_ nn.Model = &LearnedPositionalEncoder{}
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
type LearnedPositionalEncoder struct {
	nn.BaseModel
	Config  Config
	Vectors []nn.Param `spago:"type:weights;scope:model"`
}

func init() {
	gob.Register(&LearnedPositionalEncoder{})
}

// New returns a new LearnedPositionalEncoder.
// TODO: PaddingIDX
func New(config Config) *LearnedPositionalEncoder {
	vectors := make([]nn.Param, config.NumEmbeddings+config.Offset)
	for i := 0; i < len(vectors); i++ {
		vectors[i] = nn.NewParam(mat.NewEmptyVecDense[mat.Float](config.EmbeddingDim))
	}
	return &LearnedPositionalEncoder{
		Config:  config,
		Vectors: vectors,
	}
}

// Encode performs the forward step for each input and returns the result.
func (m *LearnedPositionalEncoder) Encode(positions []int) []ag.Node {
	g := m.Graph()
	embeddings := make([]ag.Node, len(positions))
	for i, pos := range positions {
		embeddings[i] = g.NewWrap(m.Vectors[pos+m.Config.Offset])
	}
	return embeddings
}
