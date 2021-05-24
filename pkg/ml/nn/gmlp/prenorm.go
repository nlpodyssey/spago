// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gmlp

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
)

var _ nn.Model = &PreNorm{}

// PreNorm is a helper model to perform pre-normalization.
type PreNorm struct {
	nn.BaseModel
	Block *Block
	Norm  *layernorm.Model
}

func init() {
	gob.Register(&PreNorm{})
}

// NewPreNorm returns a new PreNorm.
func NewPreNorm(dim int, block *Block) *PreNorm {
	return &PreNorm{
		Block: block,
		Norm:  layernorm.New(dim),
	}
}

// Forward performs the forward step.
func (m *PreNorm) Forward(xs ...ag.Node) []ag.Node {
	ns := m.Norm.Forward(xs...)
	return m.Block.Forward(ns...)
}
