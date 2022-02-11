// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package decoder

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/config"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/decoder/layer"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/positionalencoder"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/positionalencoder/learnedpositionalencoder"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/positionalencoder/sinusoidalpositionalencoder"
)

var (
	_ nn.Model[float32] = &Model[float32]{}
)

// Model implements a BART decoder.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Config             config.Config[T]
	PositionalEncoder  positionalencoder.Encoder[T]
	Layers             []*layer.Layer[T]
	EmbeddingLayerNorm *layernorm.Model[T]
	LayerNorm          *layernorm.Model[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new BART decoder Model.
func New[T mat.DType](config config.Config[T]) *Model[T] {
	return &Model[T]{
		Config:             config,
		PositionalEncoder:  newPositionalEncoder(config),
		EmbeddingLayerNorm: layernorm.New[T](config.DModel),
		Layers:             makeLayers[T](config),
		LayerNorm:          layernorm.New[T](config.DModel),
	}
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
			PaddingIDX:    0, // TODO
			Offset:        config.ExtraPosEmbedding,
		})
}

func makeLayers[T mat.DType](config config.Config[T]) []*layer.Layer[T] {
	layers := make([]*layer.Layer[T], config.DecoderLayers)
	for i := range layers {
		layers[i] = layer.NewLayer[T](config)
		// TODO: add LayerDrop to skip layers during training (see https://arxiv.org/abs/1909.11556 for description)
	}
	return layers
}

// KeysValuesPairs contains the layer.KeysValuesPairs for each decoding layer.
type KeysValuesPairs[T mat.DType] []layer.KeysValuesPairs[T]

func getPastSequenceLength[T mat.DType](pkv KeysValuesPairs[T]) int {
	if pkv == nil {
		return 0
	}
	return len(pkv[0].SelfAttKeyValues[0].Values)
}

// Decode performs the forward step for each input and returns the result.
func (m *Model[T]) Decode(
	xs []ag.Node[T],
	encoderHiddenStates []ag.Node[T],
	pastKeysValuesPairs KeysValuesPairs[T],
) ([]ag.Node[T], KeysValuesPairs[T]) {

	embedPos := m.PositionalEncoder.Encode(makePositions(len(xs), getPastSequenceLength(pastKeysValuesPairs)))
	ys := m.add(xs, embedPos)

	if m.Config.NormalizeEmbedding {
		ys = m.EmbeddingLayerNorm.Forward(ys...)
	}
	// TODO: ys = m.Dropout(ys)

	var nextCache KeysValuesPairs[T]
	for i, l := range m.Layers {
		var kvp layer.KeysValuesPairs[T]
		if pastKeysValuesPairs != nil {
			ys, kvp = l.Forward(ys, encoderHiddenStates, pastKeysValuesPairs[i])
		} else {
			ys, kvp = l.Forward(ys, encoderHiddenStates, layer.KeysValuesPairs[T]{})
		}
		nextCache = append(nextCache, kvp)
	}

	if m.Config.FinalLayerNorm {
		ys = m.LayerNorm.Forward(ys...)
	}
	return ys, nextCache
}

// makePositions returns a slice of the given size, where each element has
// the same value of its own index position plus the offset.
func makePositions(size, offset int) []int {
	indices := make([]int, size)
	for i := range indices {
		indices[i] = i + offset
	}
	return indices
}

func (m *Model[T]) add(a []ag.Node[T], b []ag.Node[T]) []ag.Node[T] {
	c := make([]ag.Node[T], len(a))
	for i := 0; i < len(a); i++ {
		c[i] = m.Graph().Add(a[i], b[i])
	}
	return c
}
