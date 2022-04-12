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

var _ nn.Model[float32] = &CrossAttention[float32]{}

// CrossAttention wraps Model to perform multi-head cross-attention, where query, key and values belong to different sequences.
type CrossAttention[T mat.DType] struct {
	*Model[T]
}

func init() {
	gob.Register(&CrossAttention[float32]{})
	gob.Register(&CrossAttention[float64]{})
}

// Forward performs the forward step for each input node and returns the result.
func (m *CrossAttention[T]) Forward(cache Cache[T], seq1 []ag.Node[T], seq2 []ag.Node[T]) ([]ag.Node[T], [][]ag.Node[T], Cache[T]) {
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
			ca := selfattention.CrossAttention[T]{Model: h}
			attentions[i], weights[i], nextCache[i] = ca.Forward(cache.At(i), seq1, seq2)
		}()
	}
	wg.Wait()

	projected := m.project(attentions, len(seq1))

	return projected, weights, nextCache
}
