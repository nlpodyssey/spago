// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multiheadattention

import (
	"encoding/gob"
	"math"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/initializers"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/mat/rand"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/attention/selfattention"
	"github.com/nlpodyssey/spago/nn/linear"
)

var _ nn.Model = &Model{}

// Model contains the serializable parameters.
type Model struct {
	nn.Module
	Heads       []*selfattention.Model
	OutputMerge *linear.Model
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model with parameters initialized to zeros.
func New[T float.DType](size, numOfHeads int, useCausalMask, isCrossAttention bool) *Model {
	return &Model{
		Heads:       makeAttentionHeads[T](size, numOfHeads, useCausalMask, isCrossAttention),
		OutputMerge: linear.New[T](size, size),
	}
}

// Init initializes the self-attention heads and the merge layer with uniform Xavier random distribution.
func (m *Model) Init(rng *rand.LockedRand) {
	gain := initializers.Gain(activation.Identity)
	initializers.XavierUniform(m.OutputMerge.W.Value().(mat.Matrix), gain, rng)
	for _, h := range m.Heads {
		h.Init(rng)
	}
}

func makeAttentionHeads[T float.DType](dm, n int, useCausalMask, isCrossAttention bool) []*selfattention.Model {
	heads := make([]*selfattention.Model, n)
	dk := dm / n
	scaleFactor := 1.0 / math.Sqrt(float64(dk))
	for i := 0; i < n; i++ {
		heads[i] = selfattention.New[T](selfattention.Config{
			InputSize:        dm,
			QuerySize:        dk,
			KeySize:          dk,
			ValueSize:        dk,
			ScaleFactor:      scaleFactor,
			UseCausalMask:    useCausalMask,
			IsCrossAttention: isCrossAttention,
		})
	}
	return heads
}

// Cache contains the self-attention cache for each head.
type Cache []selfattention.Cache

func (r Cache) At(i int) selfattention.Cache {
	if len(r) == 0 {
		return selfattention.Cache{}
	}
	return r[i]
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(cache Cache, q, x []mat.Tensor) ([]mat.Tensor, [][]mat.Tensor, Cache) {
	n := len(m.Heads)
	attentions := make([][]mat.Tensor, n)
	weights := make([][]mat.Tensor, n)
	nextCache := make(Cache, n)

	for i, h := range m.Heads {
		attentions[i], weights[i], nextCache[i] = h.Forward(cache.At(i), q, x)
	}

	projected := m.project(attentions, len(q))

	return projected, weights, nextCache
}

func (m *Model) project(heads [][]mat.Tensor, seqLen int) []mat.Tensor {
	n := len(heads)
	buf := make([]mat.Tensor, seqLen*n)
	concat := buf[:0] // shares the same backing array with buf

	for i := 0; i < seqLen; i++ {
		buf2 := buf[i*n : i*n+n]
		for j := 0; j < n; j++ {
			buf2[j] = heads[j][i]
		}
		concat = append(concat, ag.Concat(buf2...))
	}
	return m.OutputMerge.Forward(concat...)
}
