// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gmlp

import (
	"encoding/gob"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/initializers"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
)

type SpatialGatingUnit struct {
	nn.BaseModel
	Causal bool
	Norm   *layernorm.Model
	Proj   *SimpleConv1D
	Attn   *Attention
}

var _ nn.Model = &SpatialGatingUnit{}

// SpatialGatingUnitConfig provides configuration parameters for SpatialGatingUnit.
type SpatialGatingUnitConfig struct {
	Dim    int
	DimSeq int
	// Set AttnDim <= 0 to disable attention.
	AttnDim int
	Causal  bool
	InitEps mat.Float
}

func init() {
	gob.Register(&SpatialGatingUnit{})
}

// NewSpatialGatingUnit returns a new SpatialGatingUnit.
func NewSpatialGatingUnit(config SpatialGatingUnitConfig) *SpatialGatingUnit {
	dimOut := config.Dim / 2

	m := &SpatialGatingUnit{
		Causal: config.Causal,
		Norm:   layernorm.New(dimOut),
		Proj: NewSimpleConv1D(SimpleConv1DConfig{
			InputChannels:  config.DimSeq,
			OutputChannels: config.DimSeq,
		}),
		Attn: nil, // set below
	}

	if config.AttnDim > 0 {
		m.Attn = NewAttention(AttentionConfig{
			DimIn:    config.Dim,
			DimOut:   dimOut,
			DimInner: config.AttnDim,
			Causal:   config.Causal,
		})
	}

	// TODO: is it right to initialize here? Probably not...
	m.initializeProj(config.InitEps / mat.Float(config.DimSeq))

	return m
}

func (m *SpatialGatingUnit) initializeProj(initEps mat.Float) {
	r := rand.NewLockedRand(0) // TODO: random seed?
	initializers.Uniform(m.Proj.W.Value(), -initEps, initEps, r)
	initializers.Constant(m.Proj.B.Value(), 1)
}

func (m *SpatialGatingUnit) Forward(xs ...ag.Node) []ag.Node {
	g := m.Graph()
	halfSize := xs[0].Value().Size() / 2

	res := make([]ag.Node, len(xs))
	gate := make([]ag.Node, len(xs))
	for i, x := range xs {
		res[i] = g.View(x, 0, 0, halfSize, 1)
		gate[i] = g.View(x, halfSize, 0, halfSize, 1)
	}

	gate = m.Norm.Forward(gate...)

	if m.Causal {
		panic("SpatialGatingUnit.Forward causal mask is not implemented")
	} else {
		gate = m.Proj.Forward(gate...)
	}

	if m.Attn != nil {
		attn := m.Attn.Forward(xs...)
		for i, a := range attn {
			gate[i] = g.Add(gate[i], a)
		}
	}

	y := make([]ag.Node, len(gate))
	for i := range y {
		y[i] = g.Prod(gate[i], res[i])
	}
	return y
}
