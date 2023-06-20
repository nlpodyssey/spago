// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embedding

import (
	"encoding/gob"
	"log"
	"sync"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.ParamsTraverser = &Model{}

// Model implements a simple lookup table that stores fixed-size embeddings
// for a predefined dictionary. It is commonly used to store and retrieve word
// embeddings using their corresponding indices.
type Model struct {
	nn.Module
	Size         int
	Dim          int
	Weights      []*nn.Param
	embedGradIdx map[int]struct{}
	mu           sync.Mutex
}

func init() {
	gob.Register(&Model{})
}

// New returns a new embeddings Model.
func New[T float.DType](size int, dim int) *Model {
	data := make([]T, size*dim)
	weights := make([]*nn.Param, size)
	for i := 0; i < size; i++ {
		weights[i] = nn.NewParam(mat.NewDense[T](mat.WithBacking(data[i*dim : (i+1)*dim])))
	}
	return &Model{
		Size:         size,
		Dim:          dim,
		Weights:      weights,
		embedGradIdx: make(map[int]struct{}),
	}
}

// TraverseParams allows embeddings with gradients to be traversed for optimization.
func (m *Model) TraverseParams(callback func(param *nn.Param)) {
	for idx := range m.embedGradIdx {
		callback(m.Weights[idx])
	}
}

func (m *Model) Embedding(idx int) (*Embedding, error) {
	if idx < 0 || idx >= m.Size {
		return nil, nn.ErrInvalidIndex
	}
	return &Embedding{
		Param: m.Weights[idx],
		m:     m,
		idx:   idx,
	}, nil
}

// Encode returns the embedding values associated with the input indices.
// It returns an error if one of the input elements is out of range.
func (m *Model) Encode(input []int) ([]mat.Tensor, error) {
	if err := m.checkInput(input); err != nil {
		return nil, err
	}
	encoded := make([]mat.Tensor, len(input))
	for i, idx := range input {
		encoded[i] = &Embedding{
			Param: m.Weights[idx],
			m:     m,
			idx:   idx,
		}
	}
	return encoded, nil
}

// MustEncode returns the embedding values associated with the input indices.
func (m *Model) MustEncode(input []int) []mat.Tensor {
	encoded, err := m.Encode(input)
	if err != nil {
		log.Fatal(err)
	}
	return encoded
}

// checkInput returns an error if one of the input elements is out of range.
func (m *Model) checkInput(input []int) error {
	for _, idx := range input {
		if idx < 0 || idx >= m.Size {
			return nn.ErrInvalidIndex
		}
	}
	return nil
}

func (m *Model) CountEmbedWithGrad() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.embedGradIdx)
}

func (m *Model) ZeroGrad() {
	m.mu.Lock()
	defer m.mu.Unlock()
	for idx := range m.embedGradIdx {
		m.Weights[idx].ZeroGrad()
		delete(m.embedGradIdx, idx)
	}
}
