// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package selfattention

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/initializers"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/attention"
	"github.com/nlpodyssey/spago/nn/linear"
)

var _ nn.Model[float32] = &Model[float32]{}

// Cache contains the projected keys and values at index 0, 1 respectively.
type Cache[T mat.DType] [2]ag.Node[T]

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

// Init initializes the query, key and value linear layers with uniform Xavier random distribution.
func (m *Model[T]) Init(rng *rand.LockedRand[T]) {
	gain := initializers.Gain[T](activation.Identity)
	initializers.XavierUniform(m.Query.W.Value(), gain, rng)
	initializers.XavierUniform(m.Key.W.Value(), gain, rng)
	initializers.XavierUniform(m.Value.W.Value(), gain, rng)
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(cache Cache[T], q, k, v []ag.Node[T]) ([]ag.Node[T], []ag.Node[T], Cache[T]) {
	pq := m.Query.Forward(q...)

	fwKeys := m.Key.Forward(k...)
	fwValues := m.Value.Forward(v...)

	var pk, pv ag.Node[T]
	if cache[0] == nil {
		pk = ag.Stack[T](fwKeys...)
		pv = ag.Stack[T](fwValues...)
	} else {
		pk = ag.AppendRows[T](cache[0], fwKeys...)
		pv = ag.AppendRows[T](cache[1], fwValues...)
	}

	result, weights := attention.ScaledDotProductAttention(pq, pk, pv, m.ScaleFactor, m.UseCausalMask)

	return result, weights, Cache[T]{pk, pv}
}
