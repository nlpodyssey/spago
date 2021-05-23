// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gmlp

import (
	"encoding/gob"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

type Attention struct {
	nn.BaseModel
	Scale  mat.Float
	Causal bool
	ToQKV  *linear.Model
	ToOut  *linear.Model
}

var _ nn.Model = &Attention{}

// AttentionConfig provides configuration parameters for Attention.
type AttentionConfig struct {
	DimIn    int
	DimOut   int
	DimInner int
	Causal   bool
}

func init() {
	gob.Register(&Attention{})
}

// NewAttention returns a new Attention.
func NewAttention(config AttentionConfig) *Attention {
	return &Attention{
		Scale:  mat.Pow(mat.Float(config.DimInner), -0.5),
		Causal: config.Causal,
		ToQKV:  linear.New(config.DimIn, config.DimInner*3, linear.BiasGrad(false)), // TODO: verify the option
		ToOut:  linear.New(config.DimInner, config.DimOut),
	}
}

// TODO: implement Attention.Forward
func (m *Attention) Forward(xs ...ag.Node) []ag.Node {
	panic("Attention.Forward is not implemented")
}
