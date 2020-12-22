// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package posembeddings

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model     = &LearnedPositionalEmbeddings{}
	_ nn.Processor = &LearnedPositionalEmbeddingsProcessor{}
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
	Config  Config
	Vectors []nn.Param
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

// LearnedPositionalEmbeddingsProcessor implements a nn.Processor for a BART LearnedPositionalEmbeddings.
type LearnedPositionalEmbeddingsProcessor struct {
	nn.BaseProcessor
}

// NewProc returns a new processor to execute the forward step.
func (m *LearnedPositionalEmbeddings) NewProc(ctx nn.Context) nn.Processor {
	return &LearnedPositionalEmbeddingsProcessor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
	}
}

// Encode performs the forward step for each input and returns the result.
func (p *LearnedPositionalEmbeddingsProcessor) Encode(positions []int) []ag.Node {
	m := p.Model.(*LearnedPositionalEmbeddings)
	embeddings := make([]ag.Node, len(positions))
	for i, pos := range positions {
		embeddings[i] = p.Graph.NewWrap(m.Vectors[pos+m.Config.Offset])
	}
	return embeddings
}

// Forward is not implemented for LearnedPositionalEmbeddingsProcessor (it always panics).
// You should use Process instead.
func (p LearnedPositionalEmbeddingsProcessor) Forward(xs ...ag.Node) []ag.Node {
	panic("posembeddings: Forward() not implemented for LearnedPositionalEmbeddings. Use Encode() instead.")
}
