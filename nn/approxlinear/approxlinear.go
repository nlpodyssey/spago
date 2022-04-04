// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package approxlinear

import (
	"encoding/gob"

	"github.com/nlpodyssey/gomaddness"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/linear"
)

var _ nn.Model[float32] = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	*linear.Model[T]
	Maddness  *gomaddness.Maddness[T]
	CollectFn func(x mat.Matrix[T])
	UseApprox bool
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New creates a new Model.
func New[T mat.DType](linear *linear.Model[T]) *Model[T] {
	return &Model[T]{
		Model:     linear,
		UseApprox: false,
		Maddness:  nil,
		CollectFn: nil,
	}
}

func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	if m.UseApprox {
		return m.forwardApprox(xs...)
	}
	return m.forwardLinear(xs...)
}

func (m *Model[T]) forwardApprox(xs ...ag.Node[T]) []ag.Node[T] {
	g := m.Session.Graph()

	ys := make([]ag.Node[T], len(xs))
	for i, x := range xs {
		wx := g.NewOperator(NewAMM[T](m.Maddness, x))
		ys[i] = ag.Sum[T](wx, m.B)
	}
	return ys
}

func (m *Model[T]) forwardLinear(xs ...ag.Node[T]) []ag.Node[T] {
	if m.CollectFn != nil {
		for _, x := range xs {
			m.CollectFn(x.Value().Clone())
		}
	}
	return m.Model.Forward(xs...)
}
