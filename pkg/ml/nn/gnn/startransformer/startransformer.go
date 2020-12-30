// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package startransformer provides a variant implementation of the Star-Transformer model
// introduced by Qipeng Guo, Xipeng Qiu et al. in "Star-Transformer", 2019
// (https://www.aclweb.org/anthology/N19-1133.pdf).
// In this implementation, the Scaled Dot Product Attention is replaced by a Linear Attention.
package startransformer

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"sync"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	Config
	Query         *linear.Model
	Key           *linear.Model
	Value         *linear.Model
	RelayQuery    *linear.Model
	RelayKey      *linear.Model
	RelayValue    *linear.Model
	SatelliteNorm *layernorm.Model
	RelayNorm     *layernorm.Model
}

// Config provides configuration settings for a Star-Transformer Model.
type Config struct {
	InputSize int
	QuerySize int
	KeySize   int
	ValueSize int
	Steps     int
}

// New returns a new model with parameters initialized to zeros.
func New(config Config) *Model {
	return &Model{
		Config:        config,
		Query:         linear.New(config.InputSize, config.QuerySize),
		Key:           linear.New(config.InputSize, config.KeySize),
		Value:         linear.New(config.InputSize, config.ValueSize),
		RelayQuery:    linear.New(config.InputSize, config.QuerySize),
		RelayKey:      linear.New(config.InputSize, config.KeySize),
		RelayValue:    linear.New(config.InputSize, config.ValueSize),
		SatelliteNorm: layernorm.New(config.InputSize),
		RelayNorm:     layernorm.New(config.InputSize),
	}
}

// Forward performs the forward step returns the results.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	h := m.copy(xs)         // `h` are the satellite nodes
	s := m.Graph().Mean(xs) // `s` is the relay node

	for t := 1; t <= m.Steps; t++ {
		h = m.updateSatelliteNodes(h, s, xs)
		s = m.updateRelayNode(s, h)
	}
	return append(h, s)
}

func (m *Model) copy(xs []ag.Node) []ag.Node {
	g := m.Graph()
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = g.Identity(x)
	}
	return ys
}

func (m *Model) updateSatelliteNodes(prevH []ag.Node, prevS ag.Node, residual []ag.Node) []ag.Node {
	g := m.Graph()
	n := len(prevH)
	var wg sync.WaitGroup
	wg.Add(n)
	h := make([]ag.Node, n)
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
			context := []ag.Node{prevH[j], prevH[i], prevH[k], residual[i], prevS}
			h[i] = m.satelliteAttention(prevH[i], context)
			h[i] = nn.ToNode(m.SatelliteNorm.Forward(g.ReLU(h[i])))
		}(i, j, k)
	}
	wg.Wait()
	return h
}

func (m *Model) satelliteAttention(query ag.Node, context []ag.Node) ag.Node {
	attIn := attention.QKV{
		Queries: m.Query.Forward(query),
		Keys:    m.Key.Forward(context...),
		Values:  m.Value.Forward(context...),
	}
	return nn.ToNode(attention.LinearAttention(m.Graph(), attIn, attMappingFunc, 1e-12))
}

func (m *Model) updateRelayNode(prevS ag.Node, ht []ag.Node) ag.Node {
	context := append([]ag.Node{prevS}, ht...)
	s := m.relayAttention(prevS, context)
	return nn.ToNode(m.RelayNorm.Forward(m.Graph().ReLU(s)))
}

func (m *Model) relayAttention(query ag.Node, context []ag.Node) ag.Node {
	attIn := attention.QKV{
		Queries: m.RelayQuery.Forward(query),
		Keys:    m.RelayKey.Forward(context...),
		Values:  m.RelayValue.Forward(context...),
	}
	return attention.LinearAttention(m.Graph(), attIn, attMappingFunc, 1e-12)[0]
}

func attMappingFunc(g *ag.Graph, x ag.Node) ag.Node {
	return g.PositiveELU(x)
}
