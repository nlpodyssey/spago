// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sgu implements the Spatial Gating Unit (SGU).
// Reference: `Pay Attention to MLPs` by Liu et al, 2021 (https://arxiv.org/pdf/2105.08050.pdf)
package sgu

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/initializers"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/convolution/conv1x1"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
)

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	Config Config
	Norm   *layernorm.Model
	Proj   *conv1x1.Model
	Act    *activation.Model
}

var _ nn.Model = &Model{}

// Config provides configuration parameters for Model.
type Config struct {
	Dim        int
	DimSeq     int
	InitEps    float64
	Activation activation.Name
}

func init() {
	gob.Register(&Model{})
}

// New returns a new Model initialized to zeros.
func New[T mat.DType](config Config) *Model {
	dimOut := config.Dim / 2

	m := &Model{
		Config: config,
		Norm:   layernorm.New[T](dimOut, 1e-12),
		Proj: conv1x1.New[T](conv1x1.Config{
			InputChannels:  config.DimSeq,
			OutputChannels: config.DimSeq,
		}),
		Act: nil,
	}

	if config.Activation != activation.Identity {
		m.Act = activation.New(config.Activation)
	}

	return m
}

// Initialize set the projection weights as near-zero values and the biases as ones to improve training stability.
func (m *Model) Initialize(seed uint64) {
	r := rand.NewLockedRand(seed)
	eps := m.Config.InitEps / float64(m.Config.DimSeq)
	initializers.Uniform(m.Proj.W.Value(), -eps, eps, r)
	initializers.Constant(m.Proj.B.Value(), 1)
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	size := xs[0].Value().Size()
	halfSize := size / 2

	res := make([]ag.Node, len(xs))
	gate := make([]ag.Node, len(xs))
	for i, x := range xs {
		res[i] = ag.Slice(x, 0, 0, halfSize, 1)
		gate[i] = ag.Slice(x, halfSize, 0, size, 1)
	}

	gate = m.Norm.Forward(gate...)
	gate = m.Proj.Forward(gate...)

	if m.Act != nil {
		gate = m.Act.Forward(gate...)
	}

	y := make([]ag.Node, len(gate))
	for i := range y {
		y[i] = ag.Prod(gate[i], res[i])
	}
	return y
}
