// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package selfattention

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/initializers"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/mat/rand"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/attention"
	"github.com/nlpodyssey/spago/nn/linear"
)

var _ nn.Model = &Model{}

// Cache contains the projected keys and values at index 0, 1 respectively.
type Cache [2]mat.Tensor

// HasValues reports whether both values in Cache are not nil.
func (c Cache) HasValues() bool {
	return c[0] != nil && c[1] != nil
}

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	Config
	Query       *linear.Model
	Key         *linear.Model
	Value       *linear.Model
	ScaleFactor *nn.Buffer
}

// Config provides configuration settings for a Self-Attention Model.
type Config struct {
	InputSize        int
	QuerySize        int
	KeySize          int
	ValueSize        int
	ScaleFactor      float64
	UseCausalMask    bool
	IsCrossAttention bool
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](config Config) *Model {
	return &Model{
		Config:      config,
		Query:       linear.New[T](config.InputSize, config.QuerySize),
		Key:         linear.New[T](config.InputSize, config.KeySize),
		Value:       linear.New[T](config.InputSize, config.ValueSize),
		ScaleFactor: nn.Buf(mat.Scalar(T(config.ScaleFactor))),
	}
}

// Init initializes the query, key and value linear layers with uniform Xavier random distribution.
func (m *Model) Init(rng *rand.LockedRand) {
	gain := initializers.Gain(activation.Identity)
	initializers.XavierUniform(m.Query.W.Value().(mat.Matrix), gain, rng)
	initializers.XavierUniform(m.Key.W.Value().(mat.Matrix), gain, rng)
	initializers.XavierUniform(m.Value.W.Value().(mat.Matrix), gain, rng)
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(cache Cache, q, x []mat.Tensor) ([]mat.Tensor, []mat.Tensor, Cache) {
	var pk, pv mat.Tensor

	pq := m.Query.Forward(q...)

	if hasCache := cache.HasValues(); hasCache && m.IsCrossAttention {
		pk = cache[0]
		pv = cache[1]
	} else {
		k := m.Key.Forward(x...)
		v := m.Value.Forward(x...)

		if hasCache {
			pk = ag.AppendRows(cache[0], k...)
			pv = ag.AppendRows(cache[1], v...)
		} else {
			pk = ag.Stack(k...)
			pv = ag.Stack(v...)
		}
	}

	result, weights := attention.ScaledDotProductAttention(pq, pk, pv, m.ScaleFactor, m.UseCausalMask)

	return result, weights, Cache{pk, pv}
}
