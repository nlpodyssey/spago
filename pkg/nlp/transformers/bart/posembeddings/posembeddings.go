// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package posembeddings

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model = &LearnedPositionalEmbeddings{}
)

// Config provides configuration settings for a LearnedPositionalEmbeddings Model.
type Config struct {
	NumEmbeddings int
	EmbeddingDim  int
	PaddingIDX    int
	Offset        int
}

// LearnedPositionalEmbeddings contains positional embeddings fine-tuned during
// the training phase.
type LearnedPositionalEmbeddings struct {
	nn.BaseModel
	Config  Config
	Vectors []nn.Param `spago:"type:weights;scope:model"`
}

// NewLearnedPositionalEmbeddings returns a new LearnedPositionalEmbeddings.
// TODO: PaddingIDX
func NewLearnedPositionalEmbeddings(config Config) *LearnedPositionalEmbeddings {
	vectors := make([]nn.Param, config.NumEmbeddings+config.Offset)
	for i := 0; i < len(vectors); i++ {
		vectors[i] = nn.NewParam(mat.NewEmptyVecDense(config.EmbeddingDim))
	}
	return &LearnedPositionalEmbeddings{
		Config:  config,
		Vectors: vectors,
	}
}

// Encode performs the forward step for each input and returns the result.
func (m *LearnedPositionalEmbeddings) Encode(positions []int) []ag.Node {
	g := m.Graph()
	embeddings := make([]ag.Node, len(positions))
	for i, pos := range positions {
		embeddings[i] = g.NewWrap(m.Vectors[pos+m.Config.Offset])
	}
	return embeddings
}
