// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rla provides an implementation of RLA (Recurrent Linear Attention).
// See: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" by Katharopoulos et al., 2020.
// TODO: support arbitrary mapping functions
package rla

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

type Config struct {
	InputSize int
}

type Model struct {
	Config
	Wk *nn.Param `type:"weights"`
	Bk *nn.Param `type:"biases"`
	Wv *nn.Param `type:"weights"`
	Bv *nn.Param `type:"biases"`
	Wq *nn.Param `type:"weights"`
	Bq *nn.Param `type:"biases"`
}

func New(config Config) *Model {
	return &Model{
		Config: config,
		Wk:     nn.NewParam(mat.NewEmptyDense(config.InputSize, config.InputSize)),
		Bk:     nn.NewParam(mat.NewEmptyVecDense(config.InputSize)),
		Wv:     nn.NewParam(mat.NewEmptyDense(config.InputSize, config.InputSize)),
		Bv:     nn.NewParam(mat.NewEmptyVecDense(config.InputSize)),
		Wq:     nn.NewParam(mat.NewEmptyDense(config.InputSize, config.InputSize)),
		Bq:     nn.NewParam(mat.NewEmptyVecDense(config.InputSize)),
	}
}

type State struct {
	S ag.Node
	Z ag.Node
	Y ag.Node
}

type Processor struct {
	nn.BaseProcessor
	wK     ag.Node
	bK     ag.Node
	wV     ag.Node
	bV     ag.Node
	wQ     ag.Node
	bQ     ag.Node
	States []*State
}

func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	g := ctx.Graph
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: false,
		},
		States: nil,
		wK:     g.NewWrap(m.Wk),
		bK:     g.NewWrap(m.Bk),
		wV:     g.NewWrap(m.Wv),
		bV:     g.NewWrap(m.Bv),
		wQ:     g.NewWrap(m.Wq),
		bQ:     g.NewWrap(m.Bq),
	}
}

func (p *Processor) SetInitialState(state *State) {
	if len(p.States) > 0 {
		log.Fatal("lstm: the initial state must be set before any input")
	}
	p.States = append(p.States, state)
}

// Forward performs the forward step for each input and returns the result.
func (p *Processor) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		s := p.forward(x)
		p.States = append(p.States, s)
		ys[i] = s.Y
	}
	return ys
}

func (p *Processor) LastState() *State {
	n := len(p.States)
	if n == 0 {
		return nil
	}
	return p.States[n-1]
}

func (p *Processor) forward(x ag.Node) (s *State) {
	g := p.Graph
	s = new(State)

	key := nn.Affine(g, p.bK, p.wK, x)
	value := nn.Affine(g, p.bV, p.wV, x)
	query := nn.Affine(g, p.bQ, p.wQ, x)

	attKey := defaultMappingFunction(g, key)
	attQuery := defaultMappingFunction(g, query)

	if prevState := p.LastState(); prevState != nil {
		s.S = g.Add(prevState.S, g.Mul(attKey, g.T(value)))
		s.Z = g.Add(prevState.Z, attKey)
	} else {
		s.S = g.Mul(attKey, g.T(value))
		s.Z = attKey
	}

	s.Y = g.DivScalar(g.T(g.Mul(g.T(attQuery), s.S)), g.AddScalar(g.Dot(attQuery, s.Z), g.Constant(1e-12)))
	return
}

// ELU(x) + 1
func defaultMappingFunction(g *ag.Graph, x ag.Node) ag.Node {
	return g.PositiveELU(x)
}
