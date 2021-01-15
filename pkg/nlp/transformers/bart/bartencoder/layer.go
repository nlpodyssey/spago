// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bartencoder

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention/multiheadattention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/bartconfig"
)

var (
	_ nn.Model = &Layer{}
)

// Layer implements a BART encoder layer.
type Layer struct {
	nn.BaseModel
	Config                 bartconfig.Config
	SelfAttention          *multiheadattention.Model
	SelfAttentionLayerNorm *layernorm.Model
	FFN                    *stack.Model
	LayerNorm              *layernorm.Model
}

func init() {
	gob.Register(&Layer{})
}

// NewLayer returns a new BART encoder Layer.
func NewLayer(config bartconfig.Config) *Layer {
	return &Layer{
		Config:                 config,
		SelfAttention:          multiheadattention.New(config.DModel, config.EncoderAttentionHeads, false), // TODO: config.AttentionDropout
		SelfAttentionLayerNorm: layernorm.New(config.DModel),
		FFN: stack.New(
			linear.New(config.DModel, config.EncoderFFNDim),
			activation.New(ag.OpGELU), // TODO: config.ActivationFunction
			// dropout.New(config.ActivationDropout)
			linear.New(config.EncoderFFNDim, config.DModel),
			// dropout.New(config.Dropout)
		),
		LayerNorm: layernorm.New(config.DModel),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Layer) Forward(xs ...ag.Node) []ag.Node {
	selfAtt := m.selfAttentionBlock(xs)
	out := m.fullyConnectedBlock(selfAtt)
	// TODO: limit output values if any Inf or NaN
	return out
}

func (m *Layer) selfAttentionBlock(xs []ag.Node) []ag.Node {
	residual := m.copy(xs)
	if m.Config.NormalizeBefore {
		xs = m.SelfAttentionLayerNorm.Forward(xs...)
	}
	xs = m.SelfAttention.Forward(attention.ToQKV(xs)) // TODO: key_padding_mask
	// TODO: xs = m.Dropout(xs) // config.Dropout
	xs = add(m.Graph(), residual, xs)
	if !m.Config.NormalizeBefore {
		xs = m.SelfAttentionLayerNorm.Forward(xs...)
	}
	return xs
}

func (m *Layer) fullyConnectedBlock(xs []ag.Node) []ag.Node {
	residual := m.copy(xs)
	if m.Config.NormalizeBefore {
		xs = m.LayerNorm.Forward(xs...)
	}
	xs = m.FFN.Forward(xs...)
	xs = add(m.Graph(), residual, xs)
	if !m.Config.NormalizeBefore {
		xs = m.LayerNorm.Forward(xs...)
	}
	return xs
}

func (m *Layer) copy(xs []ag.Node) []ag.Node {
	g := m.Graph()
	copied := func(x ag.Node) ag.Node {
		return g.Identity(x)
	}
	return ag.Map(copied, xs)
}
