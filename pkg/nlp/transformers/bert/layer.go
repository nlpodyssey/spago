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
	_ nn.Model     = &EncoderLayer{}
	_ nn.Processor = &EncoderLayerProcessor{}
)

type EncoderLayer struct {
	MultiHeadAttention *multiheadattention.Model
	NormAttention      *layernorm.Model
	FFN                *stack.Model
	NormFFN            *layernorm.Model
	Index              int // layer index (useful for debugging)
}

type EncoderLayerProcessor struct {
	nn.BaseProcessor
	MultiHeadAttention *multiheadattention.Processor
	NormAttention      *layernorm.Processor
	FFN                *stack.Processor
	NormFFN            *layernorm.Processor
}

func (m *EncoderLayer) NewProc(ctx nn.Context) nn.Processor {
	return &EncoderLayerProcessor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		MultiHeadAttention: m.MultiHeadAttention.NewProc(ctx).(*multiheadattention.Processor),
		NormAttention:      m.NormAttention.NewProc(ctx).(*layernorm.Processor),
		FFN:                m.FFN.NewProc(ctx).(*stack.Processor),
		NormFFN:            m.NormFFN.NewProc(ctx).(*layernorm.Processor),
	}
}

func (p *EncoderLayerProcessor) Forward(xs ...ag.Node) []ag.Node {
	subLayer1 := rc.PostNorm(p.Graph, p.MultiHeadAttention.Forward, p.NormAttention.Forward, xs...)
	subLayer2 := rc.PostNorm(p.Graph, p.FFN.Forward, p.NormFFN.Forward, subLayer1...)
	return subLayer2
}
