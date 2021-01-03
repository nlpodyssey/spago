// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rla provides an implementation of RLA (Recurrent Linear Attention).
// See: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" by Katharopoulos et al., 2020.
// TODO: support arbitrary mapping functions
package rla

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
)

var (
	_ nn.Model = &Model{}
)

// Config provides configuration settings for a RLA Model.
type Config struct {
	InputSize int
}

// Model contains the serializable parameters for an RLA neural network.
type Model struct {
	nn.BaseModel
	Config
	Wk     nn.Param `spago:"type:weights"`
	Bk     nn.Param `spago:"type:biases"`
	Wv     nn.Param `spago:"type:weights"`
	Bv     nn.Param `spago:"type:biases"`
	Wq     nn.Param `spago:"type:weights"`
	Bq     nn.Param `spago:"type:biases"`
	States []*State `spago:"scope:processor"`
}

// State represent a state of the RLA recurrent network.
type State struct {
	S ag.Node
	Z ag.Node
	Y ag.Node
}

// New returns a new RLA Model, initialized according to the given configuration.
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

// SetInitialState sets the initial state of the recurrent network.
// It panics if one or more states are already present.
func (m *Model) SetInitialState(state *State) {
	if len(m.States) > 0 {
		log.Fatal("lstm: the initial state must be set before any input")
	}
	m.States = append(m.States, state)
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		s := m.forward(x)
		m.States = append(m.States, s)
		ys[i] = s.Y
	}
	return ys
}

// LastState returns the last state of the recurrent network.
// It returns nil if there are no states.
func (m *Model) LastState() *State {
	n := len(m.States)
	if n == 0 {
		return nil
	}
	return m.States[n-1]
}

func (m *Model) forward(x ag.Node) (s *State) {
	g := m.Graph()
	s = new(State)

	key := nn.Affine(g, m.Bk, m.Wk, x)
	value := nn.Affine(g, m.Bv, m.Wv, x)
	query := nn.Affine(g, m.Bq, m.Wq, x)

	attKey := defaultMappingFunction(g, key)
	attQuery := defaultMappingFunction(g, query)

	if prevState := m.LastState(); prevState != nil {
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
