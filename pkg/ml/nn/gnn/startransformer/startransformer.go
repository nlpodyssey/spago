// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package startransformer provides a variant implementation of the Star-Transformer model
// introduced by Qipeng Guo, Xipeng Qiu et al. in "Star-Transformer", 2019
// (https://www.aclweb.org/anthology/N19-1133.pdf).
// In this implementation, the Scaled Dot Product Attention is replaced by a Linear Attention.
package startransformer

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"sync"
)

var (
	_ nn.Model[float32] = &Model[float32]{}
)

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Config
	Query         *linear.Model[T]
	Key           *linear.Model[T]
	Value         *linear.Model[T]
	RelayQuery    *linear.Model[T]
	RelayKey      *linear.Model[T]
	RelayValue    *linear.Model[T]
	SatelliteNorm *layernorm.Model[T]
	RelayNorm     *layernorm.Model[T]
}

// Config provides configuration settings for a Star-Transformer Model.
type Config struct {
	InputSize int
	QuerySize int
	KeySize   int
	ValueSize int
	Steps     int
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](config Config) *Model[T] {
	return &Model[T]{
		Config:        config,
		Query:         linear.New[T](config.InputSize, config.QuerySize),
		Key:           linear.New[T](config.InputSize, config.KeySize),
		Value:         linear.New[T](config.InputSize, config.ValueSize),
		RelayQuery:    linear.New[T](config.InputSize, config.QuerySize),
		RelayKey:      linear.New[T](config.InputSize, config.KeySize),
		RelayValue:    linear.New[T](config.InputSize, config.ValueSize),
		SatelliteNorm: layernorm.New[T](config.InputSize),
		RelayNorm:     layernorm.New[T](config.InputSize),
	}
}

// Forward performs the forward step returns the results.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	h := m.copy(xs)         // `h` are the satellite nodes
	s := m.Graph().Mean(xs) // `s` is the relay node

	for t := 1; t <= m.Steps; t++ {
		h = m.updateSatelliteNodes(h, s, xs)
		s = m.updateRelayNode(s, h)
	}
	return append(h, s)
}

func (m *Model[T]) copy(xs []ag.Node[T]) []ag.Node[T] {
	g := m.Graph()
	ys := make([]ag.Node[T], len(xs))
	for i, x := range xs {
		ys[i] = g.Identity(x)
	}
	return ys
}

func (m *Model[T]) updateSatelliteNodes(prevH []ag.Node[T], prevS ag.Node[T], residual []ag.Node[T]) []ag.Node[T] {
	g := m.Graph()
	n := len(prevH)
	var wg sync.WaitGroup
	wg.Add(n)
	h := make([]ag.Node[T], n)
	first := 0
	last := n - 1
	for i := 0; i < n; i++ {
		j := i - 1
		k := i + 1
		if j < first {
			j = last
		}
		if k > last {
			k = first
		}
		go func(i, j, k int) {
			defer wg.Done()
			context := []ag.Node[T]{prevH[j], prevH[i], prevH[k], residual[i], prevS}
			h[i] = m.satelliteAttention(prevH[i], context)
			h[i] = nn.ToNode[T](m.SatelliteNorm.Forward(g.ReLU(h[i])))
		}(i, j, k)
	}
	wg.Wait()
	return h
}

func (m *Model[T]) satelliteAttention(query ag.Node[T], context []ag.Node[T]) ag.Node[T] {
	attIn := attention.QKV[T]{
		Queries: m.Query.Forward(query),
		Keys:    m.Key.Forward(context...),
		Values:  m.Value.Forward(context...),
	}
	return nn.ToNode[T](attention.LinearAttention(m.Graph(), attIn, attMappingFunc[T], 1e-12))
}

func (m *Model[T]) updateRelayNode(prevS ag.Node[T], ht []ag.Node[T]) ag.Node[T] {
	context := append([]ag.Node[T]{prevS}, ht...)
	s := m.relayAttention(prevS, context)
	return nn.ToNode[T](m.RelayNorm.Forward(m.Graph().ReLU(s)))
}

func (m *Model[T]) relayAttention(query ag.Node[T], context []ag.Node[T]) ag.Node[T] {
	attIn := attention.QKV[T]{
		Queries: m.RelayQuery.Forward(query),
		Keys:    m.RelayKey.Forward(context...),
		Values:  m.RelayValue.Forward(context...),
	}
	return attention.LinearAttention(m.Graph(), attIn, attMappingFunc[T], 1e-12)[0]
}

func attMappingFunc[T mat.DType](g *ag.Graph[T], x ag.Node[T]) ag.Node[T] {
	return g.PositiveELU(x)
}
