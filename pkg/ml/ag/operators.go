// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag/fn"
	"golang.org/x/exp/rand"
	"reflect"
)

type OpName int

const (
	Identity OpName = iota
	Dropout
	AtVec
	At
	Add
	Sub
	SubScalar
	AddScalar
	ReverseSub
	Prod
	Div
	ProdScalar
	DivScalar
	Mul
	Dot
	Reshape
	MaxPooling
	View
	Vec
	T
	Square
	Pow
	Sqrt
	Tan
	Tanh
	Sigmoid
	HardSigmoid
	HardTanh
	Softsign
	ReLU
	CeLU
	ELU
	Swish
	Mish
	LeakyReLU
	SeLU
	SoftPlus
	SoftShrink
	Threshold
	Softmax
	Sin
	Cos
	Exp
	Log
	Abs
	Neg
	Reciprocal
	ReduceSum
	ReduceMean
	Concat
	Stack
)

var opNameToMethodName = map[OpName]string{
	Identity:    "Identity",
	Dropout:     "Dropout",
	AtVec:       "AtVec",
	At:          "At",
	Add:         "Add",
	Sub:         "Sub",
	SubScalar:   "SubScalar",
	AddScalar:   "AddScalar",
	ReverseSub:  "ReverseSub",
	Prod:        "Prod",
	Div:         "Div",
	ProdScalar:  "ProdScalar",
	DivScalar:   "DivScalar",
	Mul:         "Mul",
	Dot:         "Dot",
	Reshape:     "Reshape",
	MaxPooling:  "MaxPooling",
	View:        "View",
	Vec:         "Vec",
	T:           "T",
	Square:      "Square",
	Pow:         "Pow",
	Sqrt:        "Sqrt",
	Tan:         "Tan",
	Tanh:        "Tanh",
	Sigmoid:     "Sigmoid",
	HardSigmoid: "HardSigmoid",
	HardTanh:    "HardTanh",
	Softsign:    "Softsign",
	ReLU:        "ReLU",
	CeLU:        "CeLU",
	ELU:         "ELU",
	Swish:       "Swish",
	Mish:        "Mish",
	LeakyReLU:   "LeakyReLU",
	SeLU:        "SeLU",
	SoftPlus:    "SoftPlus",
	SoftShrink:  "SoftShrink",
	Threshold:   "Threshold",
	Softmax:     "Softmax",
	Sin:         "Sin",
	Cos:         "Cos",
	Exp:         "Exp",
	Log:         "Log",
	Abs:         "Abs",
	Neg:         "Neg",
	Reciprocal:  "Reciprocal",
	ReduceSum:   "ReduceSum",
	ReduceMean:  "ReduceMean",
	Concat:      "Concat",
	Stack:       "Stack",
}

// Invoke
func (g *Graph) Invoke(operator OpName, xs ...Node) Node {
	v := reflect.ValueOf(g).MethodByName(opNameToMethodName[operator])
	args := make([]reflect.Value, len(xs))
	for i, x := range xs {
		args[i] = reflect.ValueOf(x)
	}
	ret := v.Call(args)
	return ret[0].Interface().(Node)
}

// Identity
func (g *Graph) Identity(x Node) Node {
	return g.NewOperator(fn.NewIdentity(x), x)
}

// Dropout
func (g *Graph) Dropout(x Node, p float64, source rand.Source) Node {
	return g.NewOperator(fn.NewDropout(x, p, source), x)
}

// AtVec
func (g *Graph) AtVec(x Node, i int) Node {
	return g.NewOperator(fn.NewAtVec(x, i), x)
}

// At
func (g *Graph) At(x Node, i int, j int) Node {
	return g.NewOperator(fn.NewAt(x, i, j), x)
}

// Add
// The first node may be null. This help to keep the code as concise as possible e.g. during accumulation.
func (g *Graph) Add(x1 Node, x2 Node) Node {
	if x1 != nil {
		return g.NewOperator(fn.NewAdd(x1, x2), x1, x2)
	} else {
		fake := g.NewVariable(x2.Value().ZerosLike(), false)
		return g.NewOperator(fn.NewAdd(fake, x2), fake, x2)
	}
}

// Sub
func (g *Graph) Sub(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewSub(x1, x2), x1, x2)
}

// SubScalar
func (g *Graph) SubScalar(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewSubScalar(x1, x2), x1, x2)
}

// AddScalar
func (g *Graph) AddScalar(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewAddScalar(x1, x2), x1, x2)
}

// ReverseSub
func (g *Graph) ReverseSub(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewReverseSubScalar(x1, x2), x1, x2)
}

// Prod
func (g *Graph) Prod(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewProd(x1, x2), x1, x2)
}

// Div
func (g *Graph) Div(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewDiv(x1, x2), x1, x2)
}

// ProdScalar
func (g *Graph) ProdScalar(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewProdScalar(x1, x2), x1, x2)
}

// DivScalar
func (g *Graph) DivScalar(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewDivScalar(x1, x2), x1, x2)
}

// Mul
func (g *Graph) Mul(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewMul(x1, x2), x1, x2)
}

// Dot
func (g *Graph) Dot(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewDot(x1, x2), x1, x2)
}

// Reshape
func (g *Graph) Reshape(x Node, rows, columns int) Node {
	return g.NewOperator(fn.NewReshape(x, rows, columns), x)
}

// MaxPooling
func (g *Graph) MaxPooling(x Node, rows, columns int) Node {
	return g.NewOperator(fn.NewMaxPooling(x, rows, columns), x)
}

// View
func (g *Graph) View(x Node, row, column, xStride, yStride int) Node {
	return g.NewOperator(fn.NewView(x, row, column, xStride, yStride), x)
}

// Vec
func (g *Graph) Vec(x Node) Node {
	return g.NewOperator(fn.NewVec(x), x)
}

// T
func (g *Graph) T(x Node) Node {
	return g.NewOperator(fn.NewTranspose(x), x)
}

// Square
func (g *Graph) Square(x Node) Node {
	return g.NewOperator(fn.NewSquare(x), x)
}

// Pow
func (g *Graph) Pow(x Node, power float64) Node {
	return g.NewOperator(fn.NewPow(x, power), x)
}

// Sqrt
func (g *Graph) Sqrt(x Node) Node {
	return g.NewOperator(fn.NewSqrt(x), x)
}

// Tan
func (g *Graph) Tan(x Node) Node {
	return g.NewOperator(fn.NewTan(x), x)
}

// Tanh
func (g *Graph) Tanh(x Node) Node {
	return g.NewOperator(fn.NewTanh(x), x)
}

// Sigmoid
func (g *Graph) Sigmoid(x Node) Node {
	return g.NewOperator(fn.NewSigmoid(x), x)
}

// HardSigmoid
func (g *Graph) HardSigmoid(x Node) Node {
	return g.NewOperator(fn.NewHardSigmoid(x), x)
}

// HardTanh
func (g *Graph) HardTanh(x Node) Node {
	return g.NewOperator(fn.NewHardTanh(x), x)
}

// Softsign
func (g *Graph) Softsign(x Node) Node {
	return g.NewOperator(fn.NewSoftsign(x), x)
}

// ReLU
func (g *Graph) ReLU(x Node) Node {
	return g.NewOperator(fn.NewReLU(x), x)
}

// CeLU
func (g *Graph) CeLU(x Node, alpha Node) Node {
	return g.NewOperator(fn.NewCeLU(x, alpha), x, alpha)
}

// ELU
func (g *Graph) ELU(x Node, alpha Node) Node {
	return g.NewOperator(fn.NewELU(x, alpha), x, alpha)
}

// Swish
func (g *Graph) Swish(x Node, beta Node) Node {
	return g.NewOperator(fn.NewSwish(x, beta), x, beta)
}

// Mish
func (g *Graph) Mish(x Node) Node {
	return g.NewOperator(fn.NewMish(x), x)
}

// LeakyReLU
func (g *Graph) LeakyReLU(x Node, alpha Node) Node {
	return g.NewOperator(fn.NewLeakyReLU(x, alpha), x, alpha)
}

// SeLU
func (g *Graph) SeLU(x Node, alpha Node, scale Node) Node {
	return g.NewOperator(fn.NewSeLU(x, alpha, scale), x, alpha, scale)
}

// SoftPlus
func (g *Graph) SoftPlus(x Node, beta Node, threshold Node) Node {
	return g.NewOperator(fn.NewSoftPlus(x, beta, threshold), x, beta, threshold)
}

// SoftShrink
func (g *Graph) SoftShrink(x Node, lambda Node) Node {
	return g.NewOperator(fn.NewSoftShrink(x, lambda), x, lambda)
}

// Threshold
func (g *Graph) Threshold(x Node, threshold Node, k Node) Node {
	return g.NewOperator(fn.NewThreshold(x, threshold, k), x, threshold, k)
}

// Softmax
func (g *Graph) Softmax(x Node) Node {
	return g.NewOperator(fn.NewSoftmax(x), x)
}

// Sin
func (g *Graph) Sin(x Node) Node {
	return g.NewOperator(fn.NewSin(x), x)
}

// Cos
func (g *Graph) Cos(x Node) Node {
	return g.NewOperator(fn.NewCos(x), x)
}

// Exp
func (g *Graph) Exp(x Node) Node {
	return g.NewOperator(fn.NewExp(x), x)
}

// Log
func (g *Graph) Log(x Node) Node {
	return g.NewOperator(fn.NewLog(x), x)
}

// Abs
func (g *Graph) Abs(x Node) Node {
	return g.NewOperator(fn.NewAbs(x), x)
}

// Neg
func (g *Graph) Neg(x Node) Node {
	return g.NewOperator(fn.NewNeg(x), x)
}

// Reciprocal
func (g *Graph) Reciprocal(x Node) Node {
	return g.NewOperator(fn.NewReciprocal(x), x)
}

// ReduceSum
func (g *Graph) ReduceSum(x Node) Node {
	return g.NewOperator(fn.NewReduceSum(x), x)
}

// ReduceMean
func (g *Graph) ReduceMean(x Node) Node {
	return g.NewOperator(fn.NewReduceMean(x), x)
}

// Concat
func (g *Graph) Concat(xs ...Node) Node {
	return g.NewOperator(fn.NewConcat(operands(xs)), xs...)
}

// Stack
func (g *Graph) Stack(xs ...Node) Node {
	return g.NewOperator(fn.NewStack(operands(xs)), xs...)
}
