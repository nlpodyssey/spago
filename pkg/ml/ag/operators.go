// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag/fn"
	"reflect"
)

type OpName int

const (
	OpIdentity OpName = iota
	OpDropout
	OpAtVec
	OpAt
	OpAdd
	OpSub
	OpSubScalar
	OpAddScalar
	OpReverseSub
	OpProd
	OpDiv
	OpProdScalar
	OpDivScalar
	OpMul
	OpDot
	OpReshape
	OpMaxPooling
	OpView
	OpRowView
	OpColView
	OpVec
	OpT
	OpSquare
	OpPow
	OpSqrt
	OpTan
	OpTanh
	OpSigmoid
	OpHardSigmoid
	OpHardTanh
	OpSoftsign
	OpReLU
	OpCeLU
	OpELU
	OpSwish
	OpMish
	OpLeakyReLU
	OpSeLU
	OpSoftPlus
	OpSoftShrink
	OpThreshold
	OpSoftmax
	OpSin
	OpCos
	OpExp
	OpLog
	OpAbs
	OpNeg
	OpReciprocal
	OpMax
	OpMin
	OpReduceSum
	OpReduceMean
	OpConcat
	OpStack
)

var opNameToMethodName = map[OpName]string{
	OpIdentity:    "Identity",
	OpDropout:     "Dropout",
	OpAtVec:       "AtVec",
	OpAt:          "At",
	OpAdd:         "Add",
	OpSub:         "Sub",
	OpSubScalar:   "SubScalar",
	OpAddScalar:   "AddScalar",
	OpReverseSub:  "ReverseSub",
	OpProd:        "Prod",
	OpDiv:         "Div",
	OpProdScalar:  "ProdScalar",
	OpDivScalar:   "DivScalar",
	OpMul:         "Mul",
	OpDot:         "Dot",
	OpReshape:     "Reshape",
	OpMaxPooling:  "MaxPooling",
	OpView:        "View",
	OpRowView:     "RowView",
	OpColView:     "ColView",
	OpVec:         "Vec",
	OpT:           "T",
	OpSquare:      "Square",
	OpPow:         "Pow",
	OpSqrt:        "Sqrt",
	OpTan:         "Tan",
	OpTanh:        "Tanh",
	OpSigmoid:     "Sigmoid",
	OpHardSigmoid: "HardSigmoid",
	OpHardTanh:    "HardTanh",
	OpSoftsign:    "Softsign",
	OpReLU:        "ReLU",
	OpCeLU:        "CeLU",
	OpELU:         "ELU",
	OpSwish:       "Swish",
	OpMish:        "Mish",
	OpLeakyReLU:   "LeakyReLU",
	OpSeLU:        "SeLU",
	OpSoftPlus:    "SoftPlus",
	OpSoftShrink:  "SoftShrink",
	OpThreshold:   "Threshold",
	OpSoftmax:     "Softmax",
	OpSin:         "Sin",
	OpCos:         "Cos",
	OpExp:         "Exp",
	OpLog:         "Log",
	OpAbs:         "Abs",
	OpNeg:         "Neg",
	OpReciprocal:  "Reciprocal",
	OpMax:         "Max",
	OpMin:         "Min",
	OpReduceSum:   "ReduceSum",
	OpReduceMean:  "ReduceMean",
	OpConcat:      "Concat",
	OpStack:       "Stack",
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
func (g *Graph) Dropout(x Node, p float64) Node {
	return g.NewOperator(fn.NewDropout(x, p, g.randGen), x)
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
		fake := g.NewVariable(nil, false)
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

// Max
func (g *Graph) Max(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewMax(x1, x2), x1, x2)
}

// Min
func (g *Graph) Min(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewMin(x1, x2), x1, x2)
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

// RowView
func (g *Graph) RowView(x Node, row int) Node {
	return g.NewOperator(fn.NewRowView(x, row), x)
}

// ColView
func (g *Graph) ColView(x Node, column int) Node {
	return g.NewOperator(fn.NewColView(x, column), x)
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
	return g.NewOperator(fn.NewConcat(Operands(xs)), xs...)
}

// Stack
func (g *Graph) Stack(xs ...Node) Node {
	return g.NewOperator(fn.NewStack(Operands(xs)), xs...)
}
