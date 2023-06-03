// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package activation

import (
	"encoding/gob"
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
	var operation func(x ag.DualValue) ag.DualValue

	switch m.Activation {
	case Identity:
		operation = func(x ag.DualValue) ag.DualValue { return x }
	case Tan:
		operation = ag.Tan
	case Tanh:
		operation = ag.Tanh
	case Sigmoid:
		operation = ag.Sigmoid
	case HardSigmoid:
		operation = ag.HardSigmoid
	case HardTanh:
		operation = ag.HardTanh
	case Softsign:
		operation = ag.Softsign
	case ReLU:
		operation = ag.ReLU
	case GELU:
		operation = ag.GELU
	case PositiveELU:
		operation = ag.PositiveELU
	case Swish:
		operation = ag.Swish
	case SiLU:
		operation = ag.SiLU
	case Mish:
		operation = ag.Mish
	case Softmax:
		operation = ag.Softmax
	case LogSoftmax:
		operation = ag.LogSoftmax
	case SparseMax:
		operation = ag.SparseMax
	case CELU:
		operation = func(x ag.DualValue) ag.DualValue { return ag.CELU(x, m.Params[0]) }
	case ELU:
		operation = func(x ag.DualValue) ag.DualValue { return ag.ELU(x, m.Params[0]) }
	case SwishB:
		operation = func(x ag.DualValue) ag.DualValue { return ag.SwishB(x, m.Params[0]) }
	case LeakyReLU:
		operation = func(x ag.DualValue) ag.DualValue { return ag.LeakyReLU(x, m.Params[0]) }
	case SELU:
		operation = func(x ag.DualValue) ag.DualValue { return ag.SELU(x, m.Params[0], m.Params[1]) }
	case SoftPlus:
		operation = func(x ag.DualValue) ag.DualValue { return ag.SoftPlus(x, m.Params[0], m.Params[1]) }
	case SoftShrink:
		operation = func(x ag.DualValue) ag.DualValue { return ag.SoftShrink(x, m.Params[0]) }
	case Threshold:
		operation = func(x ag.DualValue) ag.DualValue { return ag.Threshold(x, m.Params[0], m.Params[1]) }
	default:
		log.Fatal("attention: invalid activation function")
	}

	ys := make([]ag.DualValue, len(xs))

	for i, x := range xs {
		ys[i] = operation(x)
	}

	return ys
}
