// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package act

import (
	"brillion.io/spago/pkg/ml/ag"
)

type FuncName int

const (
	Identity FuncName = iota
	ReLU
	Tan
	Tanh
	HardTanh
	Sigmoid
	HardSigmoid
	SoftSign
	SoftMax
	CeLU
	ELU
	LeakyReLU
	SeLU
	SoftPlus
	SoftShrink
	Threshold
)

func F(g *ag.Graph, f FuncName, args ...ag.Node) ag.Node {
	x := args[0]
	switch f {
	case Identity:
		return g.Identity(x)
	case Tan:
		return g.Tan(x)
	case Tanh:
		return g.Tanh(x)
	case HardTanh:
		return g.HardTanh(x)
	case Sigmoid:
		return g.Sigmoid(x)
	case HardSigmoid:
		return g.HardSigmoid(x)
	case SoftSign:
		return g.Softsign(x)
	case ReLU:
		return g.ReLU(x)
	case CeLU:
		alpha := args[1]
		return g.CeLU(x, alpha)
	case ELU:
		alpha := args[1]
		return g.ELU(x, alpha)
	case LeakyReLU:
		alpha := args[1]
		return g.LeakyReLU(x, alpha)
	case SeLU:
		alpha := args[1]
		scale := args[2]
		return g.SeLU(x, alpha, scale)
	case SoftPlus:
		beta := args[1]
		threshold := args[2]
		return g.SoftPlus(x, beta, threshold)
	case SoftShrink:
		lambda := args[1]
		return g.SoftShrink(x, lambda)
	case Threshold:
		threshold := args[1]
		k := args[2]
		return g.Threshold(x, threshold, k)
	case SoftMax:
		return g.Softmax(x)
	default:
		panic("act: activation function not available")
	}
}
