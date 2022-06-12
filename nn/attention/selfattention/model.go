// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package selfattention

import (
	"encoding/gob"
	"sync"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/initializers"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/mat/rand"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/attention"
	"github.com/nlpodyssey/spago/nn/linear"
)

var _ nn.Model = &Model{}

// Cache contains the projected keys and values at index 0, 1 respectively.
type Cache [2]ag.Node

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	Config
	Query       *linear.Model
	Key         *linear.Model
	Value       *linear.Model
	ScaleFactor *ag.Constant
}

// Config provides configuration settings for a Self-Attention Model.
type Config struct {
	InputSize     int
	QuerySize     int
	KeySize       int
	ValueSize     int
	ScaleFactor   float64
	UseCausalMask bool
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
		ScaleFactor: ag.ScalarConst(T(config.ScaleFactor)),
	}
}

// Init initializes the query, key and value linear layers with uniform Xavier random distribution.
func (m *Model) Init(rng *rand.LockedRand) {
	gain := initializers.Gain(activation.Identity)
	initializers.XavierUniform(m.Query.W.Value(), gain, rng)
	initializers.XavierUniform(m.Key.W.Value(), gain, rng)
	initializers.XavierUniform(m.Value.W.Value(), gain, rng)
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(cache Cache, q, k, v []ag.Node) ([]ag.Node, []ag.Node, Cache) {
	var pq []ag.Node
	var pk, pv ag.Node

	var wg sync.WaitGroup
	wg.Add(3)

	go func() {
		pq = m.Query.Forward(q...)
		wg.Done()
	}()

	go func() {
		fwKeys := m.Key.Forward(k...)
		if cache[0] == nil {
			pk = ag.Stack(fwKeys...)
		} else {
			pk = ag.AppendRows(cache[0], fwKeys...)
		}
		wg.Done()
	}()

	go func() {
		fwValues := m.Value.Forward(v...)
		if cache[0] == nil {
			pv = ag.Stack(fwValues...)
		} else {
			pv = ag.AppendRows(cache[1], fwValues...)
		}
		wg.Done()
	}()

	wg.Wait()
	result, weights := attention.ScaledDotProductAttention(pq, pk, pv, m.ScaleFactor, m.UseCausalMask)

	return result, weights, Cache{pk, pv}
}
