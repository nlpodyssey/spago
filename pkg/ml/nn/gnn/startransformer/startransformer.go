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
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"sync"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

// Model contains the serializable parameters.
type Model struct {
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

type Processor struct {
	nn.BaseProcessor
	query         *linear.Processor
	key           *linear.Processor
	value         *linear.Processor
	relayQuery    *linear.Processor
	relayKey      *linear.Processor
	relayValue    *linear.Processor
	satelliteNorm *layernorm.Processor
	relayNorm     *layernorm.Processor
	Steps         int
}

// NewProc returns a new processor to execute the forward step.
func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		Steps:         m.Config.Steps,
		query:         m.Query.NewProc(ctx).(*linear.Processor),
		key:           m.Key.NewProc(ctx).(*linear.Processor),
		value:         m.Value.NewProc(ctx).(*linear.Processor),
		relayQuery:    m.RelayQuery.NewProc(ctx).(*linear.Processor),
		relayKey:      m.RelayKey.NewProc(ctx).(*linear.Processor),
		relayValue:    m.RelayValue.NewProc(ctx).(*linear.Processor),
		satelliteNorm: m.SatelliteNorm.NewProc(ctx).(*layernorm.Processor),
		relayNorm:     m.RelayNorm.NewProc(ctx).(*layernorm.Processor),
	}
}

// Forward performs the forward step returns the results.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	h := p.copy(xs)       // `h` are the satellite nodes
	s := p.Graph.Mean(xs) // `s` is the relay node

	for t := 1; t <= p.Steps; t++ {
		h = p.updateSatelliteNodes(h, s, xs)
		s = p.updateRelayNode(s, h)
	}
	return append(h, s)
}

func (p *Processor) copy(xs []ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		ys[i] = p.Graph.Identity(x)
	}
	return ys
}

func (p *Processor) updateSatelliteNodes(prevH []ag.Node, prevS ag.Node, residual []ag.Node) []ag.Node {
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
			h[i] = p.satelliteAttention(prevH[i], context)
			h[i] = p.satelliteNorm.Forward(p.Graph.ReLU(h[i]))[0]
		}(i, j, k)
	}
	wg.Wait()
	return h
}

func (p *Processor) satelliteAttention(query ag.Node, context []ag.Node) ag.Node {
	q := p.query.Forward(query)
	ks := p.key.Forward(context...)
	vs := p.value.Forward(context...)
	return nn.LinearAttention(p.Graph, q, ks, vs, attMappingFunc, 1e-12)[0]
}

func (p *Processor) updateRelayNode(prevS ag.Node, ht []ag.Node) ag.Node {
	context := append([]ag.Node{prevS}, ht...)
	s := p.relayAttention(prevS, context)
	return p.relayNorm.Forward(p.Graph.ReLU(s))[0]
}

func (p *Processor) relayAttention(query ag.Node, context []ag.Node) ag.Node {
	q := p.relayQuery.Forward(query)
	ks := p.relayKey.Forward(context...)
	vs := p.relayValue.Forward(context...)
	return nn.LinearAttention(p.Graph, q, ks, vs, attMappingFunc, 1e-12)[0]
}

func attMappingFunc(g *ag.Graph, x ag.Node) ag.Node {
	return g.PositiveELU(x)
}
