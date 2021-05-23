// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gmlp

import (
	"encoding/gob"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

// SimpleConv1D is a simple 1-dimensional convolution model.
// The following values are fixed: kernel size = 1; stride = 1; padding = 0,
type SimpleConv1D struct {
	nn.BaseModel
	Config SimpleConv1DConfig
	W      nn.Param `spago:"type:weights"`
	B      nn.Param `spago:"type:biases"`
}

var _ nn.Model = &SimpleConv1D{}

type SimpleConv1DConfig struct {
	InputChannels  int
	OutputChannels int
}

func init() {
	gob.Register(&SimpleConv1D{})
}

// NewSimpleConv1D returns a new SimpleConv1D.
func NewSimpleConv1D(config SimpleConv1DConfig) *SimpleConv1D {
	return &SimpleConv1D{
		Config: config,
		W:      nn.NewParam(mat.NewEmptyDense(config.OutputChannels, config.InputChannels)),
		B:      nn.NewParam(mat.NewEmptyVecDense(config.OutputChannels)),
	}
}

// Forward performs the forward step. Each "x" is a channel.
func (m *SimpleConv1D) Forward(xs ...ag.Node) []ag.Node {
	g := m.Graph()

	xm := g.Stack(xs...)
	mm := g.Mul(m.W, xm)

	ys := make([]ag.Node, m.Config.OutputChannels)
	for outCh := range ys {
		val := g.T(g.RowView(mm, outCh))
		bias := g.AtVec(m.B, outCh)
		ys[outCh] = g.AddScalar(val, bias)
	}
	return ys
}
