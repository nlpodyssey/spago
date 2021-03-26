// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package encoder

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/config"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/encoder/layer"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/positionalencoder"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/positionalencoder/learnedpositionalencoder"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/positionalencoder/sinusoidalpositionalencoder"
	"github.com/nlpodyssey/spago/pkg/utils"
)

var (
	_ nn.Model = &Model{}
)

// Model implements a BART encoder.
type Model struct {
	nn.BaseModel
	Config             config.Config
	Layers             *stack.Model
	PositionalEncoder  positionalencoder.Encoder
	EmbeddingLayerNorm *layernorm.Model
	LayerNorm          *layernorm.Model
}

func init() {
	gob.Register(&Model{})
}

func newPositionalEncoder(config config.Config) positionalencoder.Encoder {
	if config.StaticPositionEmbeddings {
		return sinusoidalpositionalencoder.New(sinusoidalpositionalencoder.Config{
			NumEmbeddings: config.VocabSize,
			EmbeddingDim:  config.DModel,
		})
	}
	return learnedpositionalencoder.New(
		learnedpositionalencoder.Config{
			NumEmbeddings: config.VocabSize,
			EmbeddingDim:  config.DModel,
			PaddingIDX:    config.PadTokenID,
			Offset:        config.ExtraPosEmbedding,
		})
}

// New returns a new BART encoder Model.
func New(config config.Config) *Model {
	return &Model{
		Config:             config,
		PositionalEncoder:  newPositionalEncoder(config),
		EmbeddingLayerNorm: layernorm.New(config.DModel),
		Layers: stack.Make(config.EncoderLayers, func(_ int) nn.StandardModel {
			return layer.NewLayer(config)
			// add LayerDrop to skip layers during training? (see https://arxiv.org/abs/1909.11556 for description)
		}),
		LayerNorm: layernorm.New(config.DModel),
	}
}

// Encode performs the forward step for each input node and returns the result.
func (m *Model) Encode(xs []ag.Node) []ag.Node {
	embedPos := m.PositionalEncoder.Encode(utils.MakeIndices(len(xs)))
	ys := add(m.Graph(), xs, embedPos)
	if m.Config.NormalizeEmbedding {
		ys = m.EmbeddingLayerNorm.Forward(ys...)
		// TODO: ys = m.Dropout(ys)
	}
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
