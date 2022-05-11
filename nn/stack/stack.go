// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stack

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model[float32] = &Model[float32]{}

// Model contains the sub-models.
type Model[T mat.DType] struct {
	nn.Module[T]
	Layers []nn.StandardModel[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model.
func New[T mat.DType](layers ...nn.StandardModel[T]) *Model[T] {
	return &Model[T]{
		Layers: layers,
	}
}

// Make makes a new model by obtaining each layer with a callback.
func Make[T mat.DType](size int, callback func(i int) nn.StandardModel[T]) *Model[T] {
	layers := make([]nn.StandardModel[T], size)
	for i := 0; i < size; i++ {
		layers[i] = callback(i)
	}
	return New(layers...)
}

// LastLayer returns the last layer from the stack.
func (m *Model[T]) LastLayer() nn.Model[T] {
	return m.Layers[len(m.Layers)-1]
}

// Find returns all layer of type M and their index.
// It not found, it returns nil, nil.
func Find[T mat.DType, M nn.Model[T]](m *Model[T]) ([]int, []*M) {
	result := make([]*M, 0)
	idx := make([]int, 0)
	for i, l := range m.Layers {
		if l, ok := any(l).(*M); ok {
			result = append(result, l)
			idx = append(idx, i)
		}
	}
	return idx, result
}

// FindOne returns the first layer of type M and its index.
// It not found, it returns -1, nil.
func FindOne[T mat.DType, M nn.Model[T]](m *Model[T]) (int, *M) {
	for i, l := range m.Layers {
		if l, ok := any(l).(*M); ok {
			return i, l
		}
	}
	return -1, nil
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	ys := m.Layers[0].Forward(xs...)
	for i := 1; i < len(m.Layers); i++ {
		ys = m.Layers[i].Forward(ys...)
	}
	return ys
}
