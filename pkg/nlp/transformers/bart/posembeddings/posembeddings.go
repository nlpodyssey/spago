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

type Config struct {
	NumEmbeddings int
	EmbeddingDim  int
	PaddingIDX    int
	Offset        int
}

type LearnedPositionalEmbeddings struct {
	Config  Config
	Vectors []*nn.Param
}

// TODO: PaddingIDX
func NewLearnedPositionalEmbeddings(config Config) *LearnedPositionalEmbeddings {
	vectors := make([]*nn.Param, config.NumEmbeddings+config.Offset)
	for i := 0; i < len(vectors); i++ {
		vectors[i] = nn.NewParam(mat.NewEmptyVecDense(config.EmbeddingDim))
	}
	return &LearnedPositionalEmbeddings{
		Config:  config,
		Vectors: vectors,
	}
}

type LearnedPositionalEmbeddingsProcessor struct {
	nn.BaseProcessor
}

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

func (p *LearnedPositionalEmbeddingsProcessor) Encode(positions []int) []ag.Node {
	m := p.Model.(*LearnedPositionalEmbeddings)
	embeddings := make([]ag.Node, len(positions))
	for i, pos := range positions {
		embeddings[i] = p.Graph.NewWrap(m.Vectors[pos+m.Config.Offset])
	}
	return embeddings
}

func (p LearnedPositionalEmbeddingsProcessor) Forward(xs ...ag.Node) []ag.Node {
	panic("posembeddings: Forward() not implemented for LearnedPositionalEmbeddings. Use Encode() instead.")
}
