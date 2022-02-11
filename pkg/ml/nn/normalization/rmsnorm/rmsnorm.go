// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package rmsnorm implements the Root Mean Square Layer Normalization method.
//
// Reference: "Root Mean Square Layer Normalization" by Biao Zhang and Rico Sennrich (2019).
// (https://arxiv.org/pdf/1910.07467.pdf)
package rmsnorm

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model[float32] = &Model[float32]{}
)

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	W nn.Param[T] `spago:"type:weights"`
	B nn.Param[T] `spago:"type:biases"`
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](size int) *Model[T] {
	return &Model[T]{
		W: nn.NewParam[T](mat.NewEmptyVecDense[T](size)),
		B: nn.NewParam[T](mat.NewEmptyVecDense[T](size)),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	g := m.Graph()
	eps := g.Constant(1e-10)
	ys := make([]ag.Node[T], len(xs))
	for i, x := range xs {
		rms := g.Sqrt(g.ReduceMean(g.Square(x)))
		ys[i] = g.Add(g.Prod(g.DivScalar(x, g.AddScalar(rms, eps)), m.W), m.B)
	}
	return ys
}
