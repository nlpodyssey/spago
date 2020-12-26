// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/multiheadattention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/rc"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
)

var (
	_ nn.Module = &EncoderLayer{}
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

// Forward performs the forward step for each input and returns the result.
func (m *EncoderLayer) Forward(xs ...ag.Node) []ag.Node {
	subLayer1 := rc.PostNorm(m.Graph(), m.MultiHeadAttention.Forward, m.NormAttention.Forward, xs...)
	subLayer2 := rc.PostNorm(m.Graph(), m.FFN.Forward, m.NormFFN.Forward, subLayer1...)
	return subLayer2
}
