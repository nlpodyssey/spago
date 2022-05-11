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

var _ nn.Model = &CrossAttention[float32]{}

// CrossAttention wraps Model to perform self-attention, where query, key and values belong to different sequences.
type CrossAttention[T mat.DType] struct {
	*Model[T]
}

func init() {
	gob.Register(&CrossAttention[float32]{})
	gob.Register(&CrossAttention[float64]{})
}

// Forward performs the forward step.
func (m CrossAttention[T]) Forward(cache Cache[T], seq1 []ag.Node[T], seq2 []ag.Node[T]) ([]ag.Node[T], []ag.Node[T], Cache[T]) {
	return m.Model.Forward(cache, seq1, seq2, seq2)
}
