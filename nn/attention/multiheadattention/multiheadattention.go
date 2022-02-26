// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multiheadattention

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/attention/selfattention"
	"github.com/nlpodyssey/spago/nn/linear"
)

var _ nn.Model[float32] = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
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

// Cache contains the self-attention cache for each head.
type Cache[T mat.DType] []*selfattention.Cache[T]

func (r Cache[T]) At(i int) *selfattention.Cache[T] {
	if len(r) == 0 {
		return nil
	}
	return r[i]
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(cache Cache[T], xs []ag.Node[T]) ([]ag.Node[T], [][]ag.Node[T], Cache[T]) {
	return m.ForwardQKV(cache, xs, xs, xs)
}

// ForwardQKV performs the forward step for each input node and returns the result.
func (m *Model[T]) ForwardQKV(cache Cache[T], q, k, v []ag.Node[T]) ([]ag.Node[T], [][]ag.Node[T], Cache[T]) {
	headsAttention := make([][]ag.Node[T], m.NumOfHeads)
	headsWeights := make([][]ag.Node[T], m.NumOfHeads)
	headsCache := make(Cache[T], m.NumOfHeads)

	for h := range m.Attention {
		headsAttention[h], headsWeights[h], headsCache[h] = m.Attention[h].ForwardQKV(cache.At(h), q, k, v)
	}

	concatHeads := make([]ag.Node[T], len(q))
	for i := 0; i < len(concatHeads); i++ {
		buf := make([]ag.Node[T], m.NumOfHeads)
		for j := 0; j < m.NumOfHeads; j++ {
			buf[j] = headsAttention[j][i]
		}
		concatHeads[i] = ag.Concat(buf...)
	}

	projected := m.OutputMerge.Forward(concatHeads...)

	return projected, headsWeights, headsCache
}
