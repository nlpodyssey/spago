// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package activation

import (
	"encoding/gob"
	"fmt"
	"log"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

type Model struct {
	nn.Module
	Activation Name
	Params     []*nn.Param
}

func init() {
	gob.Register(&Model{})
}

func New(activation Name, params ...*nn.Param) *Model {
	return &Model{
		Activation: activation,
		Params:     params,
	}
}

func (m *Model) Forward(xs ...ag.DualValue) []ag.DualValue {
	fn, err := m.activationFunc()
	if err != nil {
		log.Fatal()
	}
	ys := make([]ag.DualValue, len(xs))
	for i, x := range xs {
		ys[i] = fn(x)
	}
	return ys
}

func (m *Model) activationFunc() (func(x ag.DualValue) ag.DualValue, error) {
	switch m.Activation {
	case Identity:
		return func(x ag.DualValue) ag.DualValue { return x }, nil
	case Tan:
		return ag.Tan, nil
	case Tanh:
		return ag.Tanh, nil
	case Sigmoid:
		return ag.Sigmoid, nil
	case HardSigmoid:
		return ag.HardSigmoid, nil
	case HardTanh:
		return ag.HardTanh, nil
	case Softsign:
		return ag.Softsign, nil
	case ReLU:
		return ag.ReLU, nil
	case GELU:
		return ag.GELU, nil
	case PositiveELU:
		return ag.PositiveELU, nil
	case Swish:
		return ag.Swish, nil
	case SiLU:
		return ag.SiLU, nil
	case Mish:
		return ag.Mish, nil
	case Softmax:
		return ag.Softmax, nil
	case LogSoftmax:
		return ag.LogSoftmax, nil
	case SparseMax:
		return ag.SparseMax, nil
	case CELU:
		return func(x ag.DualValue) ag.DualValue { return ag.CELU(x, m.Params[0]) }, nil
	case ELU:
		return func(x ag.DualValue) ag.DualValue { return ag.ELU(x, m.Params[0]) }, nil
	case SwishB:
		return func(x ag.DualValue) ag.DualValue { return ag.SwishB(x, m.Params[0]) }, nil
	case LeakyReLU:
		return func(x ag.DualValue) ag.DualValue { return ag.LeakyReLU(x, m.Params[0]) }, nil
	case SELU:
		return func(x ag.DualValue) ag.DualValue { return ag.SELU(x, m.Params[0], m.Params[1]) }, nil
	case SoftPlus:
		return func(x ag.DualValue) ag.DualValue { return ag.SoftPlus(x, m.Params[0], m.Params[1]) }, nil
	case SoftShrink:
		return func(x ag.DualValue) ag.DualValue { return ag.SoftShrink(x, m.Params[0]) }, nil
	case Threshold:
		return func(x ag.DualValue) ag.DualValue { return ag.Threshold(x, m.Params[0], m.Params[1]) }, nil
	default:
		return nil, fmt.Errorf("activation: %s not supported", activationsMap[m.Activation])
	}
}
