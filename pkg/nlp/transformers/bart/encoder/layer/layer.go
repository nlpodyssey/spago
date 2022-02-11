// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package layer

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/activation"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention/multiheadattention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
	"github.com/nlpodyssey/spago/pkg/ml/nn/normalization/layernorm"
	"github.com/nlpodyssey/spago/pkg/ml/nn/stack"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/config"
)

var (
	_ nn.Model[float32] = &Layer[float32]{}
)

// Layer implements a BART encoder layer.
type Layer[T mat.DType] struct {
	nn.BaseModel[T]
	Config                 config.Config[T]
	SelfAttention          *multiheadattention.Model[T]
	SelfAttentionLayerNorm *layernorm.Model[T]
	FFN                    *stack.Model[T]
	LayerNorm              *layernorm.Model[T]
}

func init() {
	// TODO: check if this works with generics
	gob.RegisterName("*bart.encoder.layer.LayerFloat32", &Layer[float32]{})
	gob.RegisterName("*bart.encoder.layer.LayerFloat64", &Layer[float64]{})
}

// NewLayer returns a new BART encoder Layer.
func NewLayer[T mat.DType](config config.Config[T]) *Layer[T] {
	return &Layer[T]{
		Config:                 config,
		SelfAttention:          multiheadattention.New[T](config.DModel, config.EncoderAttentionHeads, false), // TODO: config.AttentionDropout
		SelfAttentionLayerNorm: layernorm.New[T](config.DModel),
		FFN: stack.New[T](
			linear.New[T](config.DModel, config.EncoderFFNDim),
			activation.New[T](mustGetOpName(config.ActivationFunction)),
			// dropout.New(config.ActivationDropout)
			linear.New[T](config.EncoderFFNDim, config.DModel),
			// dropout.New(config.Dropout)
		),
		LayerNorm: layernorm.New[T](config.DModel),
	}
}

func mustGetOpName(str string) ag.OpName {
	value, err := ag.GetOpName(str)
	if err != nil {
		panic(err)
	}
	return value
}

// Forward performs the forward step for each input node and returns the result.
func (m *Layer[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	selfAtt := m.selfAttentionBlock(xs)
	out := m.fullyConnectedBlock(selfAtt)
	// TODO: limit output values if any Inf or NaN
	return out
}

func (m *Layer[T]) selfAttentionBlock(xs []ag.Node[T]) []ag.Node[T] {
	residual := m.copy(xs)
	if m.Config.NormalizeBefore {
		xs = m.SelfAttentionLayerNorm.Forward(xs...)
	}
	xs = m.SelfAttention.Forward(attention.ToQKV(xs)).AttOutput // TODO: key_padding_mask
	// TODO: xs = m.Dropout(xs) // config.Dropout
	xs = add(m.Graph(), residual, xs)
	if !m.Config.NormalizeBefore {
		xs = m.SelfAttentionLayerNorm.Forward(xs...)
	}
	return xs
}

func (m *Layer[T]) fullyConnectedBlock(xs []ag.Node[T]) []ag.Node[T] {
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

func (m *Layer[T]) copy(xs []ag.Node[T]) []ag.Node[T] {
	g := m.Graph()
	copied := func(x ag.Node[T]) ag.Node[T] {
		return g.Identity(x)
	}
	return ag.Map(copied, xs)
}

func add[T mat.DType](g *ag.Graph[T], a []ag.Node[T], b []ag.Node[T]) []ag.Node[T] {
	c := make([]ag.Node[T], len(a))
	for i := 0; i < len(a); i++ {
		c[i] = g.Add(a[i], b[i])
	}
	return c
}
