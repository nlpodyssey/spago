// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bartdecoder

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartconfig"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/posembeddings"
	"github.com/nlpodyssey/spago/pkg/utils"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

type Model struct {
	Config                      bartconfig.Config
	LearnedPositionalEmbeddings *posembeddings.LearnedPositionalEmbeddings
	Layers                      *stack.Model
	EmbeddingLayerNorm          *layernorm.Model
	LayerNorm                   *layernorm.Model
}

func New(config bartconfig.Config) *Model {
	if config.StaticPositionEmbeddings {
		panic("bart: static position embeddings not implemented.")
	}
	if config.ScaleEmbedding {
		panic("bart: scale embedding not implemented.")
	}
	learnedPositionalEmbeddings := make([]*nn.Param, config.MaxPositionEmbeddings+config.ExtraPosEmbedding)
	for i := 0; i < len(learnedPositionalEmbeddings); i++ {
		learnedPositionalEmbeddings[i] = nn.NewParam(mat.NewEmptyVecDense(config.DModel))
	}
	return &Model{
		Config: config,
		LearnedPositionalEmbeddings: posembeddings.NewLearnedPositionalEmbeddings(
			posembeddings.Config{
				NumEmbeddings: config.VocabSize,
				EmbeddingDim:  config.DModel,
				PaddingIDX:    0, // TODO
				Offset:        config.ExtraPosEmbedding,
			}),
		EmbeddingLayerNorm: layernorm.New(config.DModel),
		Layers: stack.Make(config.DecoderLayers, func(_ int) nn.Model {
			return NewLayer(config)
			// add LayerDrop to skip layers during training (see https://arxiv.org/abs/1909.11556 for description)
		}),
		LayerNorm: layernorm.New(config.DModel),
	}
}

type Processor struct {
	nn.BaseProcessor
	bartconfig.Config
	Layers                      *stack.Processor
	LearnedPositionalEmbeddings *posembeddings.LearnedPositionalEmbeddingsProcessor
	EmbeddingLayerNorm          *layernorm.Processor
	LayerNorm                   *layernorm.Processor
}

func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		Config:                      m.Config,
		Layers:                      m.Layers.NewProc(ctx).(*stack.Processor),
		LearnedPositionalEmbeddings: m.LearnedPositionalEmbeddings.NewProc(ctx).(*posembeddings.LearnedPositionalEmbeddingsProcessor),
		EmbeddingLayerNorm:          m.EmbeddingLayerNorm.NewProc(ctx).(*layernorm.Processor),
		LayerNorm:                   m.LayerNorm.NewProc(ctx).(*layernorm.Processor),
	}
}

func (p Processor) Decode(xs, encoded []ag.Node) []ag.Node {
	embedPos := p.LearnedPositionalEmbeddings.Encode(utils.MakeIndices(len(xs)))
	ys := p.add(xs, embedPos)
	ys = p.EmbeddingLayerNorm.Forward(ys...)
	// ys = p.Dropout(ys)

	for _, layer := range p.Layers.Layers {
		ys = layer.(*LayerProcessor).Process(ys, encoded)
		// TODO: save all hidden states into the processor to allow a later access
	}

	if p.FinalLayerNorm {
		ys = p.LayerNorm.Forward(ys...)
	}
	return ys
}

func (p *Processor) add(a []ag.Node, b []ag.Node) []ag.Node {
	c := make([]ag.Node, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = p.Graph.Add(a[i], b[i])
	}
	return c
}

func (p Processor) Forward(xs ...ag.Node) []ag.Node {
	panic("bartdecoder: Forward() not implemented; use Decode() instead.")
}
