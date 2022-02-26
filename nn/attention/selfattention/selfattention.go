// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package selfattention

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/attention"
	"github.com/nlpodyssey/spago/nn/linear"
)

var _ nn.Model[float32] = &Model[float32]{}

// Cache contains the projected keys and values at index 0, 1 respectively.
type Cache[T mat.DType] [2][]ag.Node[T]

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Config[T]
	Query *linear.Model[T]
	Key   *linear.Model[T]
	Value *linear.Model[T]
}

// Config provides configuration settings for a Self-Attention Model.
type Config[T mat.DType] struct {
	InputSize     int
	QuerySize     int
	KeySize       int
	ValueSize     int
	ScaleFactor   T
	UseCausalMask bool
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](config Config[T]) *Model[T] {
	return &Model[T]{
		Config: config,
		Query:  linear.New[T](config.InputSize, config.QuerySize),
		Key:    linear.New[T](config.InputSize, config.KeySize),
		Value:  linear.New[T](config.InputSize, config.ValueSize),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(cache *Cache[T], xs []ag.Node[T]) ([]ag.Node[T], []ag.Node[T], *Cache[T]) {
	return m.ForwardQKV(cache, xs, xs, xs)
}

// ForwardQKV performs the forward step for each input node and returns the result.
func (m *Model[T]) ForwardQKV(cache *Cache[T], q, k, v []ag.Node[T]) ([]ag.Node[T], []ag.Node[T], *Cache[T]) {
	pq := m.Query.Forward(q...)
	pk := m.Key.Forward(k...)
	pv := m.Value.Forward(v...)

	if cache != nil {
		pk = append(cache[0], pk...)
		pv = append(cache[1], pv...)
	}

	result, weights := attention.ScaledDotProductAttention(pq, pk, pv, m.ScaleFactor, m.UseCausalMask)

	return result, weights, &Cache[T]{pk, pv}
}
