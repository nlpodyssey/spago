// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/multiheadattention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var (
	_ nn.Model = &EncoderLayer{}
)

// EncoderLayer is a BERT Encoder Layer model.
type EncoderLayer struct {
	nn.BaseModel
	MultiHeadAttention *multiheadattention.Model
	NormAttention      *layernorm.Model
	FFN                *stack.Model
	NormFFN            *layernorm.Model
	Index              int // layer index (useful for debugging)
}

// Forward performs the forward step for each input node and returns the result.
func (m *EncoderLayer) Forward(in interface{}) interface{} {
	subLayer1 := m.postNorm(m.MultiHeadAttention, m.NormAttention, nn.ToNodes(in))
	subLayer2 := m.postNorm(m.FFN, m.NormFFN, subLayer1)
	return subLayer2
}

// PostNorm performs post-norm residual connections:
//    y = Norm(x + F(x))
func (m *EncoderLayer) postNorm(f nn.Model, norm nn.Model, xs []ag.Node) []ag.Node {
	return norm.Forward(m.add(xs, f.Forward(xs).([]ag.Node))).([]ag.Node)
}

func (m *EncoderLayer) add(a, b []ag.Node) []ag.Node {
	c := make([]ag.Node, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = m.Graph().Add(a[i], b[i])
	}
	return c
}
