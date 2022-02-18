// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rla provides an implementation of RLA (Recurrent Linear Attention).
// See: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" by Katharopoulos et al., 2020.
// TODO: support arbitrary mapping functions
package rla

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"log"
)

var _ nn.Model = &Model[float32]{}

// Config provides configuration settings for a RLA Model.
type Config struct {
	InputSize int
}

// Model contains the serializable parameters for an RLA neural network.
type Model[T mat.DType] struct {
	nn.BaseModel
	Config
	Wk     nn.Param[T] `spago:"type:weights"`
	Bk     nn.Param[T] `spago:"type:biases"`
	Wv     nn.Param[T] `spago:"type:weights"`
	Bv     nn.Param[T] `spago:"type:biases"`
	Wq     nn.Param[T] `spago:"type:weights"`
	Bq     nn.Param[T] `spago:"type:biases"`
	States []*State[T] `spago:"scope:processor"`
}

// State represent a state of the RLA recurrent network.
type State[T mat.DType] struct {
	S ag.Node[T]
	Z ag.Node[T]
	Y ag.Node[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new RLA Model, initialized according to the given configuration.
func New[T mat.DType](config Config) *Model[T] {
	return &Model[T]{
		Config: config,
		Wk:     nn.NewParam[T](mat.NewEmptyDense[T](config.InputSize, config.InputSize)),
		Bk:     nn.NewParam[T](mat.NewEmptyVecDense[T](config.InputSize)),
		Wv:     nn.NewParam[T](mat.NewEmptyDense[T](config.InputSize, config.InputSize)),
		Bv:     nn.NewParam[T](mat.NewEmptyVecDense[T](config.InputSize)),
		Wq:     nn.NewParam[T](mat.NewEmptyDense[T](config.InputSize, config.InputSize)),
		Bq:     nn.NewParam[T](mat.NewEmptyVecDense[T](config.InputSize)),
	}
}

// SetInitialState sets the initial state of the recurrent network.
// It panics if one or more states are already present.
func (m *Model[T]) SetInitialState(state *State[T]) {
	if len(m.States) > 0 {
		log.Fatal("lstm: the initial state must be set before any input")
	}
	m.States = append(m.States, state)
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	ys := make([]ag.Node[T], len(xs))
	for i, x := range xs {
		s := m.forward(x)
		m.States = append(m.States, s)
		ys[i] = s.Y
	}
	return ys
}

// LastState returns the last state of the recurrent network.
// It returns nil if there are no states.
func (m *Model[T]) LastState() *State[T] {
	n := len(m.States)
	if n == 0 {
		return nil
	}
	return m.States[n-1]
}

func (m *Model[T]) forward(x ag.Node[T]) (s *State[T]) {
	s = new(State[T])

	key := ag.Affine[T](m.Bk, m.Wk, x)
	value := ag.Affine[T](m.Bv, m.Wv, x)
	query := ag.Affine[T](m.Bq, m.Wq, x)

	attKey := defaultMappingFunction(key)
	attQuery := defaultMappingFunction(query)

	if prevState := m.LastState(); prevState != nil {
		s.S = ag.Add(prevState.S, ag.Mul(attKey, ag.T(value)))
		s.Z = ag.Add(prevState.Z, attKey)
	} else {
		s.S = ag.Mul(attKey, ag.T(value))
		s.Z = attKey
	}

	s.Y = ag.DivScalar(ag.T(ag.Mul(ag.T(attQuery), s.S)), ag.AddScalar(ag.Dot(attQuery, s.Z), x.Graph().Constant(1e-12)))
	return
}

// ELU(x) + 1
func defaultMappingFunction[T mat.DType](x ag.Node[T]) ag.Node[T] {
	return ag.PositiveELU(x)
}
