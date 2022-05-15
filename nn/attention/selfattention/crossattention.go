// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package selfattention

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &CrossAttention{}

// CrossAttention wraps Model to perform self-attention, where query, key and values belong to different sequences.
type CrossAttention struct {
	*Model
}

func init() {
	gob.Register(&CrossAttention{})
}

// Forward performs the forward step.
func (m CrossAttention) Forward(cache Cache, seq1 []ag.Node, seq2 []ag.Node) ([]ag.Node, []ag.Node, Cache) {
	return m.Model.Forward(cache, seq1, seq2, seq2)
}
