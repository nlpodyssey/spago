// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sgu implements the Spatial Gating Unit (SGU).
// Reference: `Pay Attention to MLPs` by Liu et al, 2021 (https://arxiv.org/pdf/2105.08050.pdf)
package sgu

import (
	"encoding/gob"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/conv1x1"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	Config Config
	Norm   *layernorm.Model
	Proj   *conv1x1.Model
}

var _ nn.Model = &Model{}

// Config provides configuration parameters for Model.
type Config struct {
	Dim     int
	DimSeq  int
	InitEps mat.Float
}

func init() {
	gob.Register(&Model{})
}

// New returns a new Model initialized to zeros.
func New(config Config) *Model {
	dimOut := config.Dim / 2

	return &Model{
		Config: config,
		Norm:   layernorm.New(dimOut),
		Proj: conv1x1.New(conv1x1.Config{
			InputChannels:  config.DimSeq,
			OutputChannels: config.DimSeq,
		}),
	}
}

// Initialize set the projection weights as near-zero values and the biases as ones to improve training stability.
func (m *Model) Initialize(seed uint64) {
	r := rand.NewLockedRand(seed)
	eps := m.Config.InitEps / mat.Float(m.Config.DimSeq)
	initializers.Uniform(m.Proj.W.Value(), -eps, eps, r)
	initializers.Constant(m.Proj.B.Value(), 1)
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	g := m.Graph()
	halfSize := xs[0].Value().Size() / 2

	res := make([]ag.Node, len(xs))
	gate := make([]ag.Node, len(xs))
	for i, x := range xs {
		res[i] = g.View(x, 0, 0, halfSize, 1)
		gate[i] = g.View(x, halfSize, 0, halfSize, 1)
	}

	gate = m.Norm.Forward(gate...)
	gate = m.Proj.Forward(gate...)

	y := make([]ag.Node, len(gate))
	for i := range y {
		y[i] = g.Prod(gate[i], res[i])
	}
	return y
}
