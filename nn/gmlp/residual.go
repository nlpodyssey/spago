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

var _ nn.Model = &Residual{}

// Residual is a helper model to perform residual connections.
type Residual struct {
	nn.Module
	PreNorm *PreNorm
}

func init() {
	gob.Register(&Residual{})
}

// NewResidual returns a new Residual.
func NewResidual(preNorm *PreNorm) *Residual {
	return &Residual{
		PreNorm: preNorm,
	}
}

// Forward performs the forward step.
func (m *Residual) Forward(xs ...mat.Tensor) []mat.Tensor {
	pns := m.PreNorm.Forward(xs...)
	ys := make([]mat.Tensor, len(pns))
	for i, pn := range pns {
		ys[i] = ag.Add(pn, xs[i])
	}
	return ys
}
