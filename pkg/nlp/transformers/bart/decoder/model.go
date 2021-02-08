// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package decoder

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/config"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/decoder/layer"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/positionalencoder"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/positionalencoder/learnedpositionalencoder"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/positionalencoder/sinusoidalpositionalencoder"
	"github.com/nlpodyssey/spago/pkg/utils"
)

var (
	_ nn.Model = &Model{}
)

// Model implements a BART decoder.
type Model struct {
	nn.BaseModel
	Config             config.Config
	PositionalEncoder  positionalencoder.Encoder
	Layers             []*layer.Layer
	EmbeddingLayerNorm *layernorm.Model
	LayerNorm          *layernorm.Model
}

func init() {
	gob.Register(&Model{})
}

// New returns a new BART decoder Model.
func New(config config.Config) *Model {
	return &Model{
		Config:             config,
		PositionalEncoder:  newPositionalEncoder(config),
		EmbeddingLayerNorm: layernorm.New(config.DModel),
		Layers:             makeLayers(config),
		LayerNorm:          layernorm.New(config.DModel),
	}
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
			PaddingIDX:    0, // TODO
			Offset:        config.ExtraPosEmbedding,
		})
}

func makeLayers(config config.Config) []*layer.Layer {
	layers := make([]*layer.Layer, config.DecoderLayers)
	for i := range layers {
		layers[i] = layer.NewLayer(config)
		// TODO: add LayerDrop to skip layers during training (see https://arxiv.org/abs/1909.11556 for description)
	}
	return layers
}

type KeysValuesPairs = []layer.KeysValuesPairs

// Decode performs the forward step for each input and returns the result.
func (m *Model) Decode(
	xs []ag.Node,
	encoderHiddenStates []ag.Node,
	pastKeysValuesPairs KeysValuesPairs,
) ([]ag.Node, KeysValuesPairs) {
	embedPos := m.PositionalEncoder.Encode(utils.MakeIndices(len(xs)))
	ys := m.add(xs, embedPos)

	if m.Config.NormalizeEmbedding {
		ys = m.EmbeddingLayerNorm.Forward(ys...)
	}
	// TODO: ys = m.Dropout(ys)

	var nextCache KeysValuesPairs
	for i, l := range m.Layers {
		var kvp layer.KeysValuesPairs
		if pastKeysValuesPairs != nil {
			ys, kvp = l.Forward(ys, encoderHiddenStates, pastKeysValuesPairs[i])
		} else {
			ys, kvp = l.Forward(ys, encoderHiddenStates, layer.KeysValuesPairs{})
		}
		nextCache = append(nextCache, kvp)
	}

	if m.Config.FinalLayerNorm {
		ys = m.LayerNorm.Forward(ys...)
	}
	return ys, nextCache
}

func (m *Model) add(a []ag.Node, b []ag.Node) []ag.Node {
	c := make([]ag.Node, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = m.Graph().Add(a[i], b[i])
	}
	return c
}
