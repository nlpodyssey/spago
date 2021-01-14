// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pooling

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model = &MaxPooling{}
)

// MaxPooling is a parameter-free model used to instantiate a new Processor.
type MaxPooling struct {
	nn.BaseModel
	Rows    int
	Columns int
}

func init() {
	gob.Register(&MaxPooling{})
}

// NewMax returns a new model.
func NewMax(rows, columns int) *MaxPooling {
	return &MaxPooling{
		Rows:    rows,
		Columns: columns,
	}
}

// Forward performs the forward step for each input node and returns the result.
// The max pooling is applied independently to each input.
func (m *MaxPooling) Forward(xs ...ag.Node) []ag.Node {
	g := m.Graph()
	pooled := func(x ag.Node) ag.Node {
		return g.MaxPooling(x, m.Rows, m.Columns)
	}
	return ag.Map(pooled, xs)
}
