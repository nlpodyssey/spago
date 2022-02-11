// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention/multiheadattention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var (
	_ nn.Model[float32] = &EncoderLayer[float32]{}
)

// EncoderLayer is a BERT Encoder Layer model.
type EncoderLayer[T mat.DType] struct {
	nn.BaseModel[T]
	MultiHeadAttention *multiheadattention.Model[T]
	NormAttention      *layernorm.Model[T]
	FFN                *stack.Model[T]
	NormFFN            *layernorm.Model[T]
	Index              int // layer index (useful for debugging)
}

func init() {
	gob.Register(&EncoderLayer[float32]{})
	gob.Register(&EncoderLayer[float64]{})
}

// Forward performs the forward step for each input node and returns the result.
func (m *EncoderLayer[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	return m.fullyConnectedBlock(m.selfAttentionBlock(xs))
}

func (m *EncoderLayer[T]) selfAttentionBlock(xs []ag.Node[T]) []ag.Node[T] {
	selfAtt := m.MultiHeadAttention.Forward(attention.ToQKV(xs)).AttOutput
	return m.NormAttention.Forward(m.add(xs, selfAtt)...)
}

func (m *EncoderLayer[T]) fullyConnectedBlock(xs []ag.Node[T]) []ag.Node[T] {
	return m.NormFFN.Forward(m.add(xs, m.FFN.Forward(xs...))...)
}

func (m *EncoderLayer[T]) add(a, b []ag.Node[T]) []ag.Node[T] {
	c := make([]ag.Node[T], len(a))
	for i := 0; i < len(a); i++ {
		c[i] = m.Graph().Add(a[i], b[i])
	}
	return c
}
