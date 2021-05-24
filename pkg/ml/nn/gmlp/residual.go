// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gmlp

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var _ nn.Model = &Residual{}

// Residual is a helper model to perform residual connections.
type Residual struct {
	nn.BaseModel
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
func (m *Residual) Forward(xs ...ag.Node) []ag.Node {
	pns := m.PreNorm.Forward(xs...)

	g := m.Graph()
	ys := make([]ag.Node, len(pns))
	for i, pn := range pns {
		ys[i] = g.Add(pn, xs[i])
	}
	return ys
}
