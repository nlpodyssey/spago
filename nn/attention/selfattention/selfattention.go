// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package selfattention

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model[float32] = &SelfAttention[float32]{}

// SelfAttention wraps Model to perform self-attention, where query, key and values belong to the same sequence.
type SelfAttention[T mat.DType] struct {
	*Model[T]
}

func init() {
	gob.Register(&SelfAttention[float32]{})
	gob.Register(&SelfAttention[float64]{})
}

// Forward performs the forward step.
func (m *SelfAttention[T]) Forward(cache *Cache[T], xs []ag.Node[T]) ([]ag.Node[T], []ag.Node[T], *Cache[T]) {
	return m.Model.Forward(cache, xs, xs, xs)
}
