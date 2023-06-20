// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flatten

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model is a parameter-free model used to instantiate a new Processor.
type Model struct {
	nn.Module
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model.
func New() *Model {
	return &Model{}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...mat.Tensor) []mat.Tensor {
	vectorized := func(x mat.Tensor) mat.Tensor {
		return ag.T(ag.Flatten(x))
	}
	return []mat.Tensor{ag.Concat(ag.Map(vectorized, xs)...)}
}
