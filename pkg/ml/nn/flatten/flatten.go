// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flatten

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
)

var (
	_ nn.Model = &Model{}
)

// Model is a parameter-free model used to instantiate a new Processor.
type Model struct {
	nn.BaseModel
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model.
// TODO: think about possible configurations
func New() *Model {
	return &Model{}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	g := m.Graph()
	vectorized := func(x ag.Node) ag.Node {
		return g.Vec(x)
	}
	return []ag.Node{g.Concat(ag.Map(vectorized, xs)...)}
}
