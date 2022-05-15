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

var _ nn.Model = &CrossAttention{}

// CrossAttention wraps Model to perform multi-head cross-attention, where query, key and values belong to different sequences.
type CrossAttention struct {
	*Model
}

func init() {
	gob.Register(&CrossAttention{})
}

// Forward performs the forward step for each input node and returns the result.
func (m *CrossAttention) Forward(cache Cache, seq1 []ag.Node, seq2 []ag.Node) ([]ag.Node, [][]ag.Node, Cache) {
	n := len(m.Heads)
	attentions := make([][]ag.Node, n)
	weights := make([][]ag.Node, n)
	nextCache := make(Cache, n)

	for i, h := range m.Heads {
		ca := selfattention.CrossAttention{Model: h}
		attentions[i], weights[i], nextCache[i] = ca.Forward(cache.At(i), seq1, seq2)
	}

	projected := m.project(attentions, len(seq1))

	return projected, weights, nextCache
}
