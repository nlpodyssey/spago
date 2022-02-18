// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gmlp

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Residual[float32]{}

// Residual is a helper model to perform residual connections.
type Residual[T mat.DType] struct {
	nn.BaseModel
	PreNorm *PreNorm[T]
}

func init() {
	gob.Register(&Residual[float32]{})
	gob.Register(&Residual[float64]{})
}

// NewResidual returns a new Residual.
func NewResidual[T mat.DType](preNorm *PreNorm[T]) *Residual[T] {
	return &Residual[T]{
		PreNorm: preNorm,
	}
}

// Forward performs the forward step.
func (m *Residual[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	pns := m.PreNorm.Forward(xs...)
	ys := make([]ag.Node[T], len(pns))
	for i, pn := range pns {
		ys[i] = ag.Add(pn, xs[i])
	}
	return ys
}
