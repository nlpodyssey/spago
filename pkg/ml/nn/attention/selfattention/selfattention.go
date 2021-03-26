// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package selfattention

import (
	"encoding/gob"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	Config
	Query *linear.Model
	Key   *linear.Model
	Value *linear.Model
}

// Config provides configuration settings for a Self-Attention Model.
type Config struct {
	InputSize     int
	QuerySize     int
	KeySize       int
	ValueSize     int
	ScaleFactor   mat.Float
	UseCausalMask bool
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New(config Config) *Model {
	return &Model{
		Config: config,
		Query:  linear.New(config.InputSize, config.QuerySize),
		Key:    linear.New(config.InputSize, config.KeySize),
		Value:  linear.New(config.InputSize, config.ValueSize),
	}
}

// Forward performs the forward step for each input node and returns the result.
// It generates the queries, keys and values from the same input xs.
func (m *Model) Forward(qkv attention.QKV) attention.Output {
	projAtt := attention.QKV{
		Queries: m.Query.Forward(qkv.Queries...),
		Keys:    m.Key.Forward(qkv.Keys...),
		Values:  m.Value.Forward(qkv.Values...),
	}
	attOutput, attWeights := attention.ScaledDotProductAttention(m.Graph(), projAtt, m.ScaleFactor, m.UseCausalMask)

	return attention.Output{
		AttOutput:  attOutput,
		AttWeights: attWeights,
		ProjKeysValues: attention.KeysValuesPair{
			Keys:   projAtt.Keys,
			Values: projAtt.Values,
		},
	}
}

// ForwardWithPastKeysValues performs the forward step for each input node and returns the result.
// It generates the queries, keys and values from the same input xs.
func (m *Model) ForwardWithPastKeysValues(qkv attention.QKV, past attention.KeysValuesPair) attention.Output {
	projAtt := attention.QKV{
		Queries: m.Query.Forward(qkv.Queries...),
		Keys:    append([]ag.Node{}, past.Keys...),   // this append is important
		Values:  append([]ag.Node{}, past.Values...), // this append is important
	}

	if qkv.Keys != nil { // the qkv.Values shall not be null as well
		projAtt.Keys = append(projAtt.Keys, m.Key.Forward(qkv.Keys...)...)
		projAtt.Values = append(projAtt.Values, m.Value.Forward(qkv.Values...)...)
	}

	attOutput, attWeights := attention.ScaledDotProductAttention(m.Graph(), projAtt, m.ScaleFactor, m.UseCausalMask)

	return attention.Output{
		AttOutput:  attOutput,
		AttWeights: attWeights,
		ProjKeysValues: attention.KeysValuesPair{
			Keys:   projAtt.Keys,
			Values: projAtt.Values,
		},
	}
}
