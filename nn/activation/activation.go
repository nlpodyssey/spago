// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package activation

import (
	"encoding/gob"
	"reflect"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model contains the activation operator and serializable parameters.
type Model struct {
	nn.Module
	Activation Name
	Params     []*nn.Param
}

func init() {
	gob.Register(&Model{})
}

// New returns a new model.
func New(activation Name, params ...*nn.Param) *Model {
	return &Model{
		Activation: activation,
		Params:     params,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model) Forward(xs ...ag.Node) []ag.Node {
	if m.Activation == Identity {
		return xs
	}

	operator := activationsMap[m.Activation].operator

	args := make([]reflect.Value, len(m.Params)+1)
	for i, p := range m.Params {
		args[i+1] = reflect.ValueOf(p)
	}

	ys := make([]ag.Node, len(xs))
	for i, x := range xs {
		args[0] = reflect.ValueOf(x)
		v := operator.Call(args)
		ys[i] = v[0].Interface().(ag.Node)
	}
	return ys
}
