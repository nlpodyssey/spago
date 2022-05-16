// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rla provides an implementation of RLA (Recurrent Linear Attention).
// See: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" by Katharopoulos et al., 2020.
package rla

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Config provides configuration settings for a RLA Model.
type Config struct {
	InputSize int
}

// Model contains the serializable parameters for an RLA neural network.
type Model struct {
	nn.Module
	Config
	Wk nn.Param `spago:"type:weights"`
	Bk nn.Param `spago:"type:biases"`
	Wv nn.Param `spago:"type:weights"`
	Bv nn.Param `spago:"type:biases"`
	Wq nn.Param `spago:"type:weights"`
	Bq nn.Param `spago:"type:biases"`
}

// State represent a state of the RLA recurrent network.
type State struct {
	S ag.Node
	Z ag.Node
	Y ag.Node
}

func init() {
	gob.Register(&Model{})
}

// New returns a new RLA Model, initialized according to the given configuration.
func New[T float.DType](config Config) *Model {
	return &Model{
		Config: config,
		Wk:     nn.NewParam(mat.NewEmptyDense[T](config.InputSize, config.InputSize)),
		Bk:     nn.NewParam(mat.NewEmptyVecDense[T](config.InputSize)),
		Wv:     nn.NewParam(mat.NewEmptyDense[T](config.InputSize, config.InputSize)),
		Bv:     nn.NewParam(mat.NewEmptyVecDense[T](config.InputSize)),
		Wq:     nn.NewParam(mat.NewEmptyDense[T](config.InputSize, config.InputSize)),
		Bq:     nn.NewParam(mat.NewEmptyVecDense[T](config.InputSize)),
	}
}

func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	ys := make([]ag.Node, len(xs))
	var s *State = nil
	for i, x := range xs {
		s = m.Next(s, x)
		ys[i] = s.Y
	}
	return ys
}

// Next performs a single forward step, producing a new state.
func (m *Model) Next(prevState *State, x ag.Node) (s *State) {
	s = new(State)

	key := ag.Affine(m.Bk, m.Wk, x)
	value := ag.Affine(m.Bv, m.Wv, x)
	query := ag.Affine(m.Bq, m.Wq, x)

	attKey := defaultMappingFunction(key)
	attQuery := defaultMappingFunction(query)

	if prevState != nil {
		s.S = ag.Add(prevState.S, ag.Mul(attKey, ag.T(value)))
		s.Z = ag.Add(prevState.Z, attKey)
	} else {
		s.S = ag.Mul(attKey, ag.T(value))
		s.Z = attKey
	}

	e := ag.Var(s.Z.Value().NewScalar(1e-12))
	s.Y = ag.DivScalar(ag.T(ag.Mul(ag.T(attQuery), s.S)), ag.AddScalar(ag.Dot(attQuery, s.Z), e))
	return
}

// defaultMappingFunction returns ELU(x) + 1
// TODO: support arbitrary mapping functions
func defaultMappingFunction(x ag.Node) ag.Node {
	return ag.PositiveELU(x)
}
