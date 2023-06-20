// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package conv1x1 implements a 1-dimensional 1-kernel convolution model
package conv1x1

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

// Model is a superficial depth-wise 1-dimensional convolution model.
// The following values are fixed: kernel size = 1; stride = 1; padding = 0,
type Model struct {
	nn.Module
	Config Config
	W      *nn.Param
	B      *nn.Param
}

var _ nn.Model = &Model{}

// Config provides configuration parameters for Model.
type Config struct {
	InputChannels  int
	OutputChannels int
}

func init() {
	gob.Register(&Model{})
}

// New returns a new Model.
func New[T float.DType](config Config) *Model {
	return &Model{
		Config: config,
		W:      nn.NewParam(mat.NewDense[T](mat.WithShape(config.OutputChannels, config.InputChannels))),
		B:      nn.NewParam(mat.NewDense[T](mat.WithShape(config.OutputChannels))),
	}
}

// Forward performs the forward step. Each "x" is a channel.
func (m *Model) Forward(xs ...mat.Tensor) []mat.Tensor {
	xm := ag.Stack(xs...)
	mm := ag.Mul(m.W, xm)

	ys := make([]mat.Tensor, m.Config.OutputChannels)
	for outCh := range ys {
		val := ag.T(ag.RowView(mm, outCh))
		bias := ag.At(m.B, outCh)
		ys[outCh] = ag.AddScalar(val, bias)
	}
	return ys
}
