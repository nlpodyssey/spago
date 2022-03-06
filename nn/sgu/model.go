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
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Config Config[T]
	Norm   *layernorm.Model[T]
	Proj   *conv1x1.Model[T]
	Act    *activation.Model[T]
}

var _ nn.Model[float32] = &Model[float32]{}

// Config provides configuration parameters for Model.
type Config[T mat.DType] struct {
	Dim        int
	DimSeq     int
	InitEps    T
	Activation activation.Name
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new Model initialized to zeros.
func New[T mat.DType](config Config[T]) *Model[T] {
	dimOut := config.Dim / 2

	m := &Model[T]{
		Config: config,
		Norm:   layernorm.New[T](dimOut, 1e-12),
		Proj: conv1x1.New[T](conv1x1.Config{
			InputChannels:  config.DimSeq,
			OutputChannels: config.DimSeq,
		}),
		Act: nil,
	}

	if config.Activation != activation.Identity {
		m.Act = activation.New[T](config.Activation)
	}

	return m
}

// Initialize set the projection weights as near-zero values and the biases as ones to improve training stability.
func (m *Model[T]) Initialize(seed uint64) {
	r := rand.NewLockedRand[T](seed)
	eps := m.Config.InitEps / T(m.Config.DimSeq)
	initializers.Uniform[T](m.Proj.W.Value(), -eps, eps, r)
	initializers.Constant[T](m.Proj.B.Value(), 1)
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	halfSize := xs[0].Value().Size() / 2

	res := make([]ag.Node[T], len(xs))
	gate := make([]ag.Node[T], len(xs))
	for i, x := range xs {
		res[i] = ag.View(x, 0, 0, halfSize, 1)
		gate[i] = ag.View(x, halfSize, 0, halfSize, 1)
	}

	gate = m.Norm.Forward(gate...)
	gate = m.Proj.Forward(gate...)

	if m.Act != nil {
		gate = m.Act.Forward(gate...)
	}

	y := make([]ag.Node[T], len(gate))
	for i := range y {
		y[i] = ag.Prod(gate[i], res[i])
	}
	return y
}
