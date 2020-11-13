// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// StarTransformer is a variant of the model introduced by Qipeng Guo, Xipeng Qiu et al.
// in "Star-Transformer", 2019 (https://www.aclweb.org/anthology/N19-1133.pdf).
// In this implementation, the Scaled Dot Product Attention is replaced by a Linear Attention.
package startransformer

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

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
	States        []*State
}

type State struct {
	// H are the satellite nodes
	H []ag.Node
	// S is the relay node
	S ag.Node
}

func (m *Model) NewProc(g *ag.Graph) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              nn.Training,
			Graph:             g,
			FullSeqProcessing: true,
		},
		Steps:         m.Config.Steps,
		query:         m.Query.NewProc(g).(*linear.Processor),
		key:           m.Key.NewProc(g).(*linear.Processor),
		value:         m.Value.NewProc(g).(*linear.Processor),
		relayQuery:    m.RelayQuery.NewProc(g).(*linear.Processor),
		relayKey:      m.RelayKey.NewProc(g).(*linear.Processor),
		relayValue:    m.RelayValue.NewProc(g).(*linear.Processor),
		satelliteNorm: m.SatelliteNorm.NewProc(g).(*layernorm.Processor),
		relayNorm:     m.RelayNorm.NewProc(g).(*layernorm.Processor),
		States:        nil,
	}
}

func (p *Processor) SetMode(mode nn.ProcessingMode) {
	p.Mode = mode
	nn.SetProcessingMode(mode,
		p.query, p.key, p.value,
		p.relayQuery, p.relayKey, p.relayValue,
		p.satelliteNorm, p.relayNorm,
	)
}

func (p *Processor) newInitState(xs []ag.Node) *State {
	s := new(State)
	s.S = p.Graph.Mean(xs)
	s.H = make([]ag.Node, len(xs))
	for i, x := range xs {
		s.H[i] = p.Graph.Identity(x)
	}
	return s
}

func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	p.States = append(p.States, p.newInitState(xs))

	for t := 1; t <= p.Steps; t++ {
		prev := p.States[t-1]
		h := p.updateSatelliteNodes(prev.H, prev.S, xs)
		s := p.updateRelayNode(prev.S, h)
		p.States = append(p.States, &State{H: h, S: s})
	}

	lastState := p.States[len(p.States)-1]

	return append(lastState.H, lastState.S)
}

func (p *Processor) updateSatelliteNodes(prevH []ag.Node, prevS ag.Node, residual []ag.Node) []ag.Node {
	n := len(prevH)
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
		context := []ag.Node{prevH[j], prevH[i], prevH[k], residual[i], prevS}
		h[i] = p.satelliteAttention(prevH[i], context)
		h[i] = p.satelliteNorm.Forward(p.Graph.ReLU(h[i]))[0]
	}
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
