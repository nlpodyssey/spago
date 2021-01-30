// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention/multiheadattention"
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

func init() {
	gob.Register(&EncoderLayer{})
}

// Forward performs the forward step for each input node and returns the result.
func (m *EncoderLayer) Forward(xs ...ag.Node) []ag.Node {
	return m.fullyConnectedBlock(m.selfAttentionBlock(xs))
}

func (m *EncoderLayer) selfAttentionBlock(xs []ag.Node) []ag.Node {
	selfAtt := m.MultiHeadAttention.Forward(attention.ToQKV(xs)).AttOutput
	return m.NormAttention.Forward(m.add(xs, selfAtt)...)
}

func (m *EncoderLayer) fullyConnectedBlock(xs []ag.Node) []ag.Node {
	return m.NormFFN.Forward(m.add(xs, m.FFN.Forward(xs...))...)
}

func (m *EncoderLayer) add(a, b []ag.Node) []ag.Node {
	c := make([]ag.Node, len(a))
	for i := 0; i < len(a); i++ {
		c[i] = m.Graph().Add(a[i], b[i])
	}
	return c
}
