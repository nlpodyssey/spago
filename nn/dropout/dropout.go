// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dropout

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model is a parameter-free model.
type Model struct {
	nn.Module
	P float64
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model.
func New(p float64) *Model {
	return &Model{
		P: p,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.DualValue) []ag.DualValue {
	if m.P == 0 {
		return xs
	}
	return ag.Map(ag.DropoutFunc(m.P), xs)
}
