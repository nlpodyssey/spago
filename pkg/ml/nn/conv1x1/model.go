// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package conv1x1 implements a 1-dimensional 1-kernel convolution model
package conv1x1

import (
	"encoding/gob"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

// Model is a superficial depth-wise 1-dimensional convolution model.
// The following values are fixed: kernel size = 1; stride = 1; padding = 0,
type Model struct {
	nn.BaseModel
	Config Config
	W      nn.Param `spago:"type:weights"`
	B      nn.Param `spago:"type:biases"`
}

var _ nn.Model = &Model{}

type Config struct {
	InputChannels  int
	OutputChannels int
}

func init() {
	gob.Register(&Model{})
}

// New returns a new Model.
func New(config Config) *Model {
	return &Model{
		Config: config,
		W:      nn.NewParam(mat.NewEmptyDense(config.OutputChannels, config.InputChannels)),
		B:      nn.NewParam(mat.NewEmptyVecDense(config.OutputChannels)),
	}
}

// Forward performs the forward step. Each "x" is a channel.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
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
