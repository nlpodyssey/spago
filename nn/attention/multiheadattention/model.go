// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multiheadattention

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/initializers"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/attention/selfattention"
	"github.com/nlpodyssey/spago/nn/linear"
)

var _ nn.Model[float32] = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	Heads       []*selfattention.Model[T]
	OutputMerge *linear.Model[T]
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](size, numOfHeads int, useCausalMask bool) *Model[T] {
	return &Model[T]{
		Heads:       makeAttentionHeads[T](size, numOfHeads, useCausalMask),
		OutputMerge: linear.New[T](size, size),
	}
}

// Init initializes the self-attention heads and the merge layer with uniform Xavier random distribution.
func (m *Model[T]) Init(rng *rand.LockedRand[T]) {
	gain := initializers.Gain[T](activation.Identity)
	initializers.XavierUniform(m.OutputMerge.W.Value(), gain, rng)
	for _, h := range m.Heads {
		h.Init(rng)
	}
}

func makeAttentionHeads[T mat.DType](dm, n int, useCausalMask bool) []*selfattention.Model[T] {
	heads := make([]*selfattention.Model[T], n)
	dk := dm / n
	scaleFactor := 1.0 / mat.Sqrt(T(dk))
	for i := 0; i < n; i++ {
		heads[i] = selfattention.New(selfattention.Config[T]{
			InputSize:     dm,
			QuerySize:     dk,
			KeySize:       dk,
			ValueSize:     dk,
			ScaleFactor:   scaleFactor,
			UseCausalMask: useCausalMask,
		})
	}
	return heads
}

// Cache contains the self-attention cache for each head.
type Cache[T mat.DType] []selfattention.Cache[T]

func (r Cache[T]) At(i int) selfattention.Cache[T] {
	if len(r) == 0 {
		return selfattention.Cache[T]{}
	}
	return r[i]
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(cache Cache[T], q, k, v []ag.Node[T]) ([]ag.Node[T], [][]ag.Node[T], Cache[T]) {
	n := len(m.Heads)
	attentions := make([][]ag.Node[T], n)
	weights := make([][]ag.Node[T], n)
	nextCache := make(Cache[T], n)

	for i, h := range m.Heads {
		attentions[i], weights[i], nextCache[i] = h.Forward(cache.At(i), q, k, v)
	}

	projected := m.project(attentions, len(q))

	return projected, weights, nextCache
}

func (m *Model[T]) project(heads [][]ag.Node[T], seqLen int) []ag.Node[T] {
	n := len(heads)
	concat := make([]ag.Node[T], seqLen)
	buf := make([]ag.Node[T], n*seqLen)
	for i := 0; i < seqLen; i++ {
		buf2 := buf[i*n : i*n+n]
		for j := 0; j < n; j++ {
			buf2[j] = heads[j][i]
		}
		concat[i] = ag.Concat(buf2...)
	}
	return m.OutputMerge.Forward(concat...)
}
