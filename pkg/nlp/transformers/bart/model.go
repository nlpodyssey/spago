// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bart implements the transformer model introduced by Mike et al., 2019.
// "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"
// https://arxiv.org/abs/1910.13461
package bart

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/embeddings"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/config"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/decoder"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/encoder"
	"strconv"
)

var (
	_ nn.Model[float32] = &Model[float32]{}
)

// Model implements a BART model.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Config     config.Config[T]
	Embeddings *embeddings.Model[T]
	Encoder    *encoder.Model[T]
	Decoder    *decoder.Model[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new BART Model.
func New[T mat.DType](config config.Config[T], embeddingsStoragePath string) *Model[T] {
	return &Model[T]{
		Config: config,
		Embeddings: embeddings.New[T](embeddings.Config{
			Size:       config.DModel,
			DBPath:     embeddingsStoragePath,
			ReadOnly:   !config.Training,
			ForceNewDB: false, // TODO: from config?
		}),
		Encoder: encoder.New[T](config),
		Decoder: decoder.New[T](config),
	}
}

// Close closes the BART model's embeddings DB.
func (m *Model[T]) Close() {
	m.Embeddings.Close()
}

// Process performs the forward step for each input and returns the result.
func (m *Model[T]) Process(inputIDs []int) []ag.Node[T] {
	encoded := m.Encode(inputIDs)
	decoded, _ := m.Decode(shiftR(inputIDs, 1), encoded, nil)
	return decoded
}

// Encode performs the BART encoding.
func (m *Model[T]) Encode(inputIDs []int) []ag.Node[T] {
	return m.Encoder.Encode(m.useScaledEmbeddings(m.Embeddings.Encode(intToStringSlice(inputIDs))))
}

// Decode performs the BART decoding.
func (m *Model[T]) Decode(
	inputIDs []int,
	encoderHiddenStates []ag.Node[T],
	pastKeysValuesPairs decoder.KeysValuesPairs[T],
) ([]ag.Node[T], decoder.KeysValuesPairs[T]) {
	return m.Decoder.Decode(
		m.useScaledEmbeddings(m.Embeddings.Encode(intToStringSlice(inputIDs))),
		encoderHiddenStates,
		pastKeysValuesPairs,
	)
}

func (m *Model[T]) useScaledEmbeddings(xs []ag.Node[T]) []ag.Node[T] {
	if !m.Config.ScaleEmbedding {
		return xs
	}

	embedScale := m.Graph().Constant(mat.Sqrt(T(m.Config.DModel)))
	scaled := func(x ag.Node[T]) ag.Node[T] {
		return m.Graph().ProdScalar(x, embedScale)
	}
	return ag.Map(scaled, xs)
}

func intToStringSlice(a []int) []string {
	out := make([]string, len(a))
	for i, num := range a {
		out[i] = strconv.Itoa(num)
	}
	return out
}

func shiftR(a []int, i int) []int {
	x, b := a[:(len(a)-i)], a[(len(a)-i):]
	return append(b, x...)
}
