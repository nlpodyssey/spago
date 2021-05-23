// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gmlp

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

type Residual struct {
	nn.BaseModel
	PreNorm *PreNorm
}

var _ nn.Model = &Residual{}

func init() {
	gob.Register(&Residual{})
}

// NewResidual returns a new Residual.
func NewResidual(preNorm *PreNorm) *Residual {
	return &Residual{
		PreNorm: preNorm,
	}
}

func (m *Residual) Forward(xs ...ag.Node) []ag.Node {
	pns := m.PreNorm.Forward(xs...)

	g := m.Graph()
	ys := make([]ag.Node, len(pns))
	for i, pn := range pns {
		ys[i] = g.Add(pn, xs[i])
	}
	return ys
}
