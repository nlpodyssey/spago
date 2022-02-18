// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multiheadattention

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/attention"
	"github.com/nlpodyssey/spago/nn/attention/selfattention"
	"github.com/nlpodyssey/spago/nn/linear"
)

var _ nn.Model = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel
	Attention   []*selfattention.Model[T]
	OutputMerge *linear.Model[T]
	NumOfHeads  int // number of heads
	Dm          int // input and output vectors dimension
	Dk          int // hidden vectors dimension (Dm / NumOfHeads)
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](size, numOfHeads int, useCausalMask bool) *Model[T] {
	dm := size
	dk := size / numOfHeads
	att := make([]*selfattention.Model[T], numOfHeads)
	attentionConfig := selfattention.Config[T]{
		InputSize:     dm,
		QuerySize:     dk,
		KeySize:       dk,
		ValueSize:     dk,
		ScaleFactor:   1.0 / mat.Sqrt(T(dk)),
		UseCausalMask: useCausalMask,
	}
	for i := 0; i < numOfHeads; i++ {
		att[i] = selfattention.New(attentionConfig)
	}
	return &Model[T]{
		Attention:   att,
		OutputMerge: linear.New[T](dk*numOfHeads, dm),
		NumOfHeads:  numOfHeads,
		Dm:          dm,
		Dk:          dk,
	}
}

// KeysValuesPairs contains the attention.KeysValuesPair for each attention head.
type KeysValuesPairs[T mat.DType] []attention.KeysValuesPair[T]

// Output aggregates the multiple output of the multi-head attentions,
// incl. attention scores and last projected keys and values for each head.
type Output[T mat.DType] struct {
	// Result of the multi-head attention.
	AttOutput []ag.Node[T]
	// AttWeights attention scores.
	AttWeights [][]mat.Matrix[T]
	// ProjKeysValues contains the attention.KeysValuesPair for each attention head.
	ProjKeysValues KeysValuesPairs[T]
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(qkv attention.QKV[T]) Output[T] {
	return m.forward(qkv, nil)
}

// ForwardWithPastKeysValues performs the forward step for each input node and returns the result.
func (m *Model[T]) ForwardWithPastKeysValues(qkv attention.QKV[T], pastProjKeysValues KeysValuesPairs[T]) Output[T] {
	return m.forward(qkv, pastProjKeysValues)
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) forward(qkv attention.QKV[T], pastProjKeysValues KeysValuesPairs[T]) Output[T] {
	headsAttNodes := make([][]ag.Node[T], m.NumOfHeads)
	headsAttWeights := make([][]mat.Matrix[T], m.NumOfHeads)
	attProjKeysValues := make(KeysValuesPairs[T], m.NumOfHeads)

	for h, proc := range m.Attention {
		var out attention.Output[T]
		if pastProjKeysValues != nil {
			out = proc.ForwardWithPastKeysValues(qkv, pastProjKeysValues[h])
		} else {
			out = proc.Forward(qkv)
		}
		headsAttNodes[h] = out.AttOutput
		headsAttWeights[h] = out.AttWeights
		attProjKeysValues[h] = out.ProjKeysValues
	}

	concatHeads := make([]ag.Node[T], len(qkv.Queries))
	for i := 0; i < len(concatHeads); i++ {
		buf := make([]ag.Node[T], m.NumOfHeads)
		for j := 0; j < m.NumOfHeads; j++ {
			buf[j] = headsAttNodes[j][i]
		}
		concatHeads[i] = ag.Concat(buf...)
	}

	return Output[T]{
		AttOutput:      m.OutputMerge.Forward(concatHeads...),
		AttWeights:     headsAttWeights,
		ProjKeysValues: attProjKeysValues,
	}
}
