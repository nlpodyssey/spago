// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package selfattention

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &SelfAttention{}

// SelfAttention wraps Model to perform self-attention, where query, key and values belong to the same sequence.
type SelfAttention struct {
	*Model
}

func init() {
	gob.Register(&SelfAttention{})
}

// Forward performs the forward step.
func (m SelfAttention) Forward(cache Cache, xs []ag.Node) ([]ag.Node, []ag.Node, Cache) {
	return m.Model.Forward(cache, xs, xs, xs)
}
