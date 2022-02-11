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
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/encoder/layer"
)

var (
	_ nn.Model[float32] = &layer.Layer[float32]{}
)

// Layer implements a BART decoder layer.
type Layer[T mat.DType] struct {
	nn.BaseModel[T]
	Config                    config.Config[T]
	SelfAttention             *multiheadattention.Model[T]
	SelfAttentionLayerNorm    *layernorm.Model[T]
	EncoderAttention          *multiheadattention.Model[T]
	EncoderAttentionLayerNorm *layernorm.Model[T]
	FFN                       *stack.Model[T]
	LayerNorm                 *layernorm.Model[T]
}

func init() {
	// TODO: check if this works with generics
	gob.RegisterName("*bart.decoder.layer.LayerFloat32", &Layer[float32]{})
	gob.RegisterName("*bart.decoder.layer.LayerFloat64", &Layer[float64]{})
}

// NewLayer returns a new BART decoder Layer.
func NewLayer[T mat.DType](config config.Config[T]) *Layer[T] {
	return &Layer[T]{
		Config: config,
		SelfAttention: multiheadattention.New[T](
			config.DModel,
			config.DecoderAttentionHeads,
			true, // use causal mask
			// TODO: config.AttentionDropout
		),
		SelfAttentionLayerNorm: layernorm.New[T](config.DModel),
		EncoderAttention: multiheadattention.New[T](
			config.DModel,
			config.DecoderAttentionHeads,
			false, // don't use causal mask
			// TODO: config.AttentionDropout, encoder_decoder_attention=True
		),
		EncoderAttentionLayerNorm: layernorm.New[T](config.DModel),
		FFN: stack.New[T](
			linear.New[T](config.DModel, config.DecoderFFNDim),
			activation.New[T](mustGetOpName(config.ActivationFunction)),
			// dropout.New(config.ActivationDropout)
			linear.New[T](config.DecoderFFNDim, config.DModel),
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

// KeysValuesPairs contains the keys and values used by the self-attention and cross-attention blocks.
type KeysValuesPairs[T mat.DType] struct {
	// SelfAttKeyValues contains the keys and values used by self-attention.
	SelfAttKeyValues multiheadattention.KeysValuesPairs[T]
	// CrossAttKeyValues contains the keys and values used by cross-attention.
	CrossAttKeyValues multiheadattention.KeysValuesPairs[T]
}

// Forward performs the forward step for each input and returns the result.
func (m *Layer[T]) Forward(
	xs []ag.Node[T],
	encoderHiddenStates []ag.Node[T],
	pastProjKeysValues KeysValuesPairs[T],
) ([]ag.Node[T], KeysValuesPairs[T]) {
	selfAtt, selfAttKeyValues := m.selfAttentionBlock(xs, pastProjKeysValues.SelfAttKeyValues)
	crossAtt, crossAttKeyValues := m.crossAttentionBlock(selfAtt, encoderHiddenStates, pastProjKeysValues.CrossAttKeyValues)
	out := m.fullyConnectedBlock(crossAtt)

	return out, KeysValuesPairs[T]{
		SelfAttKeyValues:  selfAttKeyValues,
		CrossAttKeyValues: crossAttKeyValues,
	}
}

func (m *Layer[T]) selfAttentionBlock(
	xs []ag.Node[T],
	pastProjKeysValues multiheadattention.KeysValuesPairs[T],
) ([]ag.Node[T], multiheadattention.KeysValuesPairs[T]) {
	residual := m.copy(xs)
	if m.Config.NormalizeBefore {
		xs = m.SelfAttentionLayerNorm.Forward(xs...)
	}
	att := m.SelfAttention.ForwardWithPastKeysValues(attention.ToQKV(xs), pastProjKeysValues)
	xs = att.AttOutput
	// TODO: xs = m.Dropout(xs)
	xs = m.add(residual, xs)
	if !m.Config.NormalizeBefore {
		xs = m.SelfAttentionLayerNorm.Forward(xs...)
	}
	return xs, att.ProjKeysValues
}

func (m *Layer[T]) crossAttentionBlock(
	xs []ag.Node[T],
	encoderHiddenStates []ag.Node[T],
	pastProjKeysValues multiheadattention.KeysValuesPairs[T],
) ([]ag.Node[T], multiheadattention.KeysValuesPairs[T]) {
	residual := m.copy(xs)
	if m.Config.NormalizeBefore {
		xs = m.EncoderAttentionLayerNorm.Forward(xs...)
	}

	qkv := attention.QKV[T]{Queries: xs}
	// use the past key-values if they are available otherwise use the encoder hidden states
	if pastProjKeysValues == nil {
		qkv.Keys = encoderHiddenStates
		qkv.Values = encoderHiddenStates
	}

	att := m.EncoderAttention.ForwardWithPastKeysValues(qkv, pastProjKeysValues)
	xs = att.AttOutput
	// TODO: xs = m.Dropout(xs)
	xs = m.add(residual, xs)
	if !m.Config.NormalizeBefore {
		xs = m.EncoderAttentionLayerNorm.Forward(xs...)
	}
	return xs, att.ProjKeysValues
}

func (m *Layer[T]) fullyConnectedBlock(xs []ag.Node[T]) []ag.Node[T] {
	residual := m.copy(xs)
	if m.Config.NormalizeBefore {
		xs = m.LayerNorm.Forward(xs...)
	}
	xs = m.FFN.Forward(xs...)
	xs = m.add(residual, xs)
	if !m.Config.NormalizeBefore {
		xs = m.LayerNorm.Forward(xs...)
	}
	return xs
}

func (m *Layer[T]) copy(xs []ag.Node[T]) []ag.Node[T] {
	copied := func(x ag.Node[T]) ag.Node[T] {
		return m.Graph().Identity(x)
	}
	return ag.Map(copied, xs)
}

func (m *Layer[T]) add(a []ag.Node[T], b []ag.Node[T]) []ag.Node[T] {
	c := make([]ag.Node[T], len(a))
	for i := 0; i < len(a); i++ {
		c[i] = m.Graph().Add(a[i], b[i])
	}
	return c
}
