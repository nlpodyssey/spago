// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multiheadattention

import (
	"encoding/gob"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/attention/selfattention"
	"github.com/nlpodyssey/spago/pkg/ml/nn/linear"
)

var (
	_ nn.Model = &Model{}
)

// Model contains the serializable parameters.
type Model struct {
	nn.BaseModel
	Attention   []*selfattention.Model
	OutputMerge *linear.Model
	NumOfHeads  int // number of heads
	Dm          int // input and output vectors dimension
	Dk          int // hidden vectors dimension (Dm / NumOfHeads)
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New(size, numOfHeads int, useCausalMask bool) *Model {
	dm := size
	dk := size / numOfHeads
	att := make([]*selfattention.Model, numOfHeads)
	attentionConfig := selfattention.Config{
		InputSize:     dm,
		QuerySize:     dk,
		KeySize:       dk,
		ValueSize:     dk,
		ScaleFactor:   1.0 / mat.Sqrt(mat.Float(dk)),
		UseCausalMask: useCausalMask,
	}
	for i := 0; i < numOfHeads; i++ {
		att[i] = selfattention.New(attentionConfig)
	}
	return &Model{
		Attention:   att,
		OutputMerge: linear.New(dk*numOfHeads, dm),
		NumOfHeads:  numOfHeads,
		Dm:          dm,
		Dk:          dk,
	}
}

// KeysValuesPairs contains the attention.KeysValuesPair for each attention head.
type KeysValuesPairs = []attention.KeysValuesPair

// Output aggregates the multiple output of the multi-head attentions,
// incl. attention scores and last projected keys and values for each head.
type Output struct {
	// Result of the multi-head attention.
	AttOutput []ag.Node
	// AttWeights attention scores.
	AttWeights [][]mat.Matrix
	// ProjKeysValues contains the attention.KeysValuesPair for each attention head.
	ProjKeysValues KeysValuesPairs
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(qkv attention.QKV) Output {
	return m.forward(qkv, nil)
}

// ForwardWithPastKeysValues performs the forward step for each input node and returns the result.
func (m *Model) ForwardWithPastKeysValues(qkv attention.QKV, pastProjKeysValues KeysValuesPairs) Output {
	return m.forward(qkv, pastProjKeysValues)
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) forward(qkv attention.QKV, pastProjKeysValues KeysValuesPairs) Output {
	headsAttNodes := make([][]ag.Node, m.NumOfHeads)
	headsAttWeights := make([][]mat.Matrix, m.NumOfHeads)
	attProjKeysValues := make(KeysValuesPairs, m.NumOfHeads)

	for h, proc := range m.Attention {
		var out attention.Output
		if pastProjKeysValues != nil {
			out = proc.ForwardWithPastKeysValues(qkv, pastProjKeysValues[h])
		} else {
			out = proc.Forward(qkv)
		}
		headsAttNodes[h] = out.AttOutput
		headsAttWeights[h] = out.AttWeights
		attProjKeysValues[h] = out.ProjKeysValues
	}

	concatHeads := make([]ag.Node, len(qkv.Queries))
	for i := 0; i < len(concatHeads); i++ {
		buf := make([]ag.Node, m.NumOfHeads)
		for j := 0; j < m.NumOfHeads; j++ {
			buf[j] = headsAttNodes[j][i]
		}
		concatHeads[i] = m.Graph().Concat(buf...)
	}

	return Output{
		AttOutput:      m.OutputMerge.Forward(concatHeads...),
		AttWeights:     headsAttWeights,
		ProjKeysValues: attProjKeysValues,
	}
}
