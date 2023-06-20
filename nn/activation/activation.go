// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package activation

import (
	"encoding/gob"
	"fmt"
	"log"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

type Model struct {
	nn.Module
	Activation Activation
	Params     []*nn.Param
}

func init() {
	gob.Register(&Model{})
}

func New(activation Activation, params ...*nn.Param) *Model {
	return &Model{
		Activation: activation,
		Params:     params,
	}
}

func (m *Model) Forward(xs ...mat.Tensor) []mat.Tensor {
	fn, err := m.activationFunc()
	if err != nil {
		log.Fatal()
	}
	ys := make([]mat.Tensor, len(xs))
	for i, x := range xs {
		ys[i] = fn(x)
	}
	return ys
}

func (m *Model) activationFunc() (func(x mat.Tensor) mat.Tensor, error) {
	if f, ok := activationFunctions[m.Activation]; ok {
		return f, nil
	}

	switch m.Activation {
	case CELU:
		return func(x mat.Tensor) mat.Tensor { return ag.CELU(x, m.Params[0]) }, nil
	case ELU:
		return func(x mat.Tensor) mat.Tensor { return ag.ELU(x, m.Params[0]) }, nil
	case SwishB:
		return func(x mat.Tensor) mat.Tensor { return ag.SwishB(x, m.Params[0]) }, nil
	case LeakyReLU:
		return func(x mat.Tensor) mat.Tensor { return ag.LeakyReLU(x, m.Params[0]) }, nil
	case SELU:
		return func(x mat.Tensor) mat.Tensor { return ag.SELU(x, m.Params[0], m.Params[1]) }, nil
	case SoftPlus:
		return func(x mat.Tensor) mat.Tensor { return ag.SoftPlus(x, m.Params[0], m.Params[1]) }, nil
	case SoftShrink:
		return func(x mat.Tensor) mat.Tensor { return ag.SoftShrink(x, m.Params[0]) }, nil
	case Threshold:
		return func(x mat.Tensor) mat.Tensor { return ag.Threshold(x, m.Params[0], m.Params[1]) }, nil
	default:
		return nil, fmt.Errorf("activation: %s not supported", activationsMap[m.Activation])
	}
}
