// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multiheadattention

import (
	"encoding/gob"
	"sync"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/attention/selfattention"
)

var _ nn.Model[float32] = &SelfAttention[float32]{}

// SelfAttention wraps Model to perform multi-head self-attention, where query, key and values belong to the same sequence.
type SelfAttention[T mat.DType] struct {
	*Model[T]
}

func init() {
	gob.Register(&SelfAttention[float32]{})
	gob.Register(&SelfAttention[float64]{})
}

// Forward performs the forward step for each input node and returns the result.
func (m *SelfAttention[T]) Forward(cache Cache[T], xs []ag.Node[T]) ([]ag.Node[T], [][]ag.Node[T], Cache[T]) {
	n := len(m.Heads)
	attentions := make([][]ag.Node[T], n)
	weights := make([][]ag.Node[T], n)
	nextCache := make(Cache[T], n)

	var wg sync.WaitGroup
	wg.Add(len(m.Heads))
	for i, h := range m.Heads {
		i, h := i, h
		go func() {
			defer wg.Done()
			sa := selfattention.SelfAttention[T]{Model: h}
			attentions[i], weights[i], nextCache[i] = sa.Forward(cache.At(i), xs)
		}()
	}
	wg.Wait()

	projected := m.project(attentions, len(xs))

	return projected, weights, nextCache
}
