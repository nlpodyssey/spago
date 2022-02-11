// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package selfattention

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

var (
	_ nn.Model[float32] = &Model[float32]{}
)

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Config[T]
	Query *linear.Model[T]
	Key   *linear.Model[T]
	Value *linear.Model[T]
}

// Config provides configuration settings for a Self-Attention Model.
type Config[T mat.DType] struct {
	InputSize     int
	QuerySize     int
	KeySize       int
	ValueSize     int
	ScaleFactor   T
	UseCausalMask bool
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](config Config[T]) *Model[T] {
	return &Model[T]{
		Config: config,
		Query:  linear.New[T](config.InputSize, config.QuerySize),
		Key:    linear.New[T](config.InputSize, config.KeySize),
		Value:  linear.New[T](config.InputSize, config.ValueSize),
	}
}

// Forward performs the forward step for each input node and returns the result.
// It generates the queries, keys and values from the same input xs.
func (m *Model[T]) Forward(qkv attention.QKV[T]) attention.Output[T] {
	projAtt := attention.QKV[T]{
		Queries: m.Query.Forward(qkv.Queries...),
		Keys:    m.Key.Forward(qkv.Keys...),
		Values:  m.Value.Forward(qkv.Values...),
	}
	attOutput, attWeights := attention.ScaledDotProductAttention(m.Graph(), projAtt, m.ScaleFactor, m.UseCausalMask)

	return attention.Output[T]{
		AttOutput:  attOutput,
		AttWeights: attWeights,
		ProjKeysValues: attention.KeysValuesPair[T]{
			Keys:   projAtt.Keys,
			Values: projAtt.Values,
		},
	}
}

// ForwardWithPastKeysValues performs the forward step for each input node and returns the result.
// It generates the queries, keys and values from the same input xs.
func (m *Model[T]) ForwardWithPastKeysValues(qkv attention.QKV[T], past attention.KeysValuesPair[T]) attention.Output[T] {
	projAtt := attention.QKV[T]{
		Queries: m.Query.Forward(qkv.Queries...),
		Keys:    append([]ag.Node[T]{}, past.Keys...),   // this append is important
		Values:  append([]ag.Node[T]{}, past.Values...), // this append is important
	}

	if qkv.Keys != nil { // the qkv.Values shall not be null as well
		projAtt.Keys = append(projAtt.Keys, m.Key.Forward(qkv.Keys...)...)
		projAtt.Values = append(projAtt.Values, m.Value.Forward(qkv.Values...)...)
	}

	attOutput, attWeights := attention.ScaledDotProductAttention(m.Graph(), projAtt, m.ScaleFactor, m.UseCausalMask)

	return attention.Output[T]{
		AttOutput:  attOutput,
		AttWeights: attWeights,
		ProjKeysValues: attention.KeysValuesPair[T]{
			Keys:   projAtt.Keys,
			Values: projAtt.Values,
		},
	}
}
