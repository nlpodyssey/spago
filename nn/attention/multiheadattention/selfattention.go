// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multiheadattention

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/attention/selfattention"
)

var _ nn.Model = &SelfAttention{}

// SelfAttention wraps Model to perform multi-head self-attention, where query, key and values belong to the same sequence.
type SelfAttention struct {
	*Model
}

func init() {
	gob.Register(&SelfAttention{})
}

// Forward performs the forward step for each input node and returns the result.
func (m *SelfAttention) Forward(cache Cache, xs []ag.DualValue) ([]ag.DualValue, [][]ag.DualValue, Cache) {
	n := len(m.Heads)
	attentions := make([][]ag.DualValue, n)
	weights := make([][]ag.DualValue, n)
	nextCache := make(Cache, n)

	for i, h := range m.Heads {
		sa := selfattention.SelfAttention{Model: h}
		attentions[i], weights[i], nextCache[i] = sa.Forward(cache.At(i), xs)
	}

	projected := m.project(attentions, len(xs))

	return projected, weights, nextCache
}
