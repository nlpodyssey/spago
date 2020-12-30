// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bartencoder

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartconfig"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/posembeddings"
	"github.com/nlpodyssey/spago/pkg/utils"
)

var (
	_ nn.Model = &Model{}
)

// Model implements a BART encoder.
type Model struct {
	nn.BaseModel
	Config                      bartconfig.Config
	Layers                      *stack.Model
	LearnedPositionalEmbeddings *posembeddings.LearnedPositionalEmbeddings
	EmbeddingLayerNorm          *layernorm.Model
	LayerNorm                   *layernorm.Model
}

// New returns a new BART encoder Model.
func New(config bartconfig.Config) *Model {
	if config.StaticPositionEmbeddings {
		panic("bart: static position embeddings not implemented.")
	}
	if config.ScaleEmbedding {
		panic("bart: scale embedding not implemented.")
	}

	return &Model{
		Config: config,
		LearnedPositionalEmbeddings: posembeddings.NewLearnedPositionalEmbeddings(
			posembeddings.Config{
				NumEmbeddings: config.VocabSize,
				EmbeddingDim:  config.DModel,
				PaddingIDX:    config.PadTokenID,
				Offset:        config.ExtraPosEmbedding,
			}),
		EmbeddingLayerNorm: layernorm.New(config.DModel),
		Layers: stack.Make(config.EncoderLayers, func(_ int) nn.StandardModel {
			return NewLayer(config)
			// add LayerDrop to skip layers during training? (see https://arxiv.org/abs/1909.11556 for description)
		}),
		LayerNorm: layernorm.New(config.DModel),
	}
}

// Encode performs the forward step for each input node and returns the result.
func (m *Model) Encode(xs []ag.Node) []ag.Node {
	embedPos := m.LearnedPositionalEmbeddings.Encode(utils.MakeIndices(len(xs)))
	ys := add(m.Graph(), xs, embedPos)
	ys = m.EmbeddingLayerNorm.Forward(ys...)
	// TODO: ys = m.Dropout(ys)

	ys = m.Layers.Forward(ys...)
	if m.Config.FinalLayerNorm {
		ys = m.LayerNorm.Forward(ys...)
	}
	return ys // TODO: return all hidden states?
}

func add(g *ag.Graph, a []ag.Node, b []ag.Node) []ag.Node {
	c := make([]ag.Node, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = g.Add(a[i], b[i])
	}
	return c
}
