// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package encoder

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
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
	_ nn.Model[float32] = &Model[float32]{}
)

// Model implements a BART encoder.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Config             config.Config[T]
	Layers             *stack.Model[T]
	PositionalEncoder  positionalencoder.Encoder[T]
	EmbeddingLayerNorm *layernorm.Model[T]
	LayerNorm          *layernorm.Model[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

func newPositionalEncoder[T mat.DType](config config.Config[T]) positionalencoder.Encoder[T] {
	if config.StaticPositionEmbeddings {
		return sinusoidalpositionalencoder.New[T](sinusoidalpositionalencoder.Config{
			NumEmbeddings: config.VocabSize,
			EmbeddingDim:  config.DModel,
		})
	}
	return learnedpositionalencoder.New[T](
		learnedpositionalencoder.Config{
			NumEmbeddings: config.VocabSize,
			EmbeddingDim:  config.DModel,
			PaddingIDX:    config.PadTokenID,
			Offset:        config.ExtraPosEmbedding,
		})
}

// New returns a new BART encoder Model.
func New[T mat.DType](config config.Config[T]) *Model[T] {
	return &Model[T]{
		Config:             config,
		PositionalEncoder:  newPositionalEncoder[T](config),
		EmbeddingLayerNorm: layernorm.New[T](config.DModel),
		Layers: stack.Make(config.EncoderLayers, func(_ int) nn.StandardModel[T] {
			return layer.NewLayer[T](config)
			// add LayerDrop to skip layers during training? (see https://arxiv.org/abs/1909.11556 for description)
		}),
		LayerNorm: layernorm.New[T](config.DModel),
	}
}

// Encode performs the forward step for each input node and returns the result.
func (m *Model[T]) Encode(xs []ag.Node[T]) []ag.Node[T] {
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

func add[T mat.DType](g *ag.Graph[T], a []ag.Node[T], b []ag.Node[T]) []ag.Node[T] {
	c := make([]ag.Node[T], len(a))
	for i := 0; i < len(a); i++ {
		c[i] = g.Add(a[i], b[i])
	}
	return c
}
