// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag/fn"
	"github.com/pkg/errors"
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
	OpGeLU
	OpELU
	OpPositiveELU
	OpSwish
	OpMish
	OpLeakyReLU
	OpSeLU
	OpSoftPlus
	OpSoftShrink
	OpThreshold
	OpSoftmax
	OpSparseMax
	OpSparseMaxLoss
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
	OpMean
	OpSum
	OpConcat
	OpStack
)

var opNameToMethodName = map[OpName]string{
	OpIdentity:      "Identity",
	OpDropout:       "Dropout",
	OpAtVec:         "AtVec",
	OpAt:            "At",
	OpAdd:           "Add",
	OpSub:           "Sub",
	OpSubScalar:     "SubScalar",
	OpAddScalar:     "AddScalar",
	OpReverseSub:    "ReverseSub",
	OpProd:          "Prod",
	OpDiv:           "Div",
	OpProdScalar:    "ProdScalar",
	OpDivScalar:     "DivScalar",
	OpMul:           "Mul",
	OpDot:           "Dot",
	OpReshape:       "Reshape",
	OpMaxPooling:    "MaxPooling",
	OpView:          "View",
	OpRowView:       "RowView",
	OpColView:       "ColView",
	OpVec:           "Vec",
	OpT:             "T",
	OpSquare:        "Square",
	OpPow:           "Pow",
	OpSqrt:          "Sqrt",
	OpTan:           "Tan",
	OpTanh:          "Tanh",
	OpSigmoid:       "Sigmoid",
	OpHardSigmoid:   "HardSigmoid",
	OpHardTanh:      "HardTanh",
	OpSoftsign:      "Softsign",
	OpReLU:          "ReLU",
	OpCeLU:          "CeLU",
	OpGeLU:          "GeLU",
	OpELU:           "ELU",
	OpPositiveELU:   "PositiveELU",
	OpSwish:         "Swish",
	OpMish:          "Mish",
	OpLeakyReLU:     "LeakyReLU",
	OpSeLU:          "SeLU",
	OpSoftPlus:      "SoftPlus",
	OpSoftShrink:    "SoftShrink",
	OpThreshold:     "Threshold",
	OpSoftmax:       "Softmax",
	OpSparseMax:     "SparseMax",
	OpSparseMaxLoss: "SparseMaxLoss",
	OpSin:           "Sin",
	OpCos:           "Cos",
	OpExp:           "Exp",
	OpLog:           "Log",
	OpAbs:           "Abs",
	OpNeg:           "Neg",
	OpReciprocal:    "Reciprocal",
	OpMax:           "Max",
	OpMin:           "Min",
	OpReduceSum:     "ReduceSum",
	OpReduceMean:    "ReduceMean",
	OpMean:          "Mean",
	OpSum:           "Sum",
	OpConcat:        "Concat",
	OpStack:         "Stack",
}

// strToOpName is the inverse map of opNameToMethodName
var strToOpName = func() map[string]OpName {
	invMap := make(map[string]OpName)
	for k, v := range opNameToMethodName {
		invMap[v] = k
	}
	return invMap
}()

// GetOpName maps a string to an operator.
// It panics if the string does not match any operator.
func GetOpName(str string) (OpName, error) {
	if value, ok := strToOpName[str]; ok {
		return value, nil
	}
	return -1, errors.Errorf("ag: unknown operator %s", str)
}

// Invoke returns a new node as a result of the application of the input operator.
func (g *Graph) Invoke(operator OpName, xs ...Node) Node {
	v := reflect.ValueOf(g).MethodByName(opNameToMethodName[operator])
	args := make([]reflect.Value, len(xs))
	for i, x := range xs {
		args[i] = reflect.ValueOf(x)
	}
	ret := v.Call(args)
	return ret[0].Interface().(Node)
}

// Identity returns a new operator node as a result of the fn.Identity function.
func (g *Graph) Identity(x Node) Node {
	return g.NewOperator(fn.NewIdentity(x), x)
}

// Dropout returns a new operator node as a result of the fn.Dropout function.
func (g *Graph) Dropout(x Node, p float64) Node {
	return g.NewOperator(fn.NewDropout(x, p, g.randGen), x)
}

// AtVec returns a new operator node as a result of the fn.AtVec function.
func (g *Graph) AtVec(x Node, i int) Node {
	return g.NewOperator(fn.NewAtVec(x, i), x)
}

// At returns a new operator node as a result of the fn.At function.
func (g *Graph) At(x Node, i int, j int) Node {
	return g.NewOperator(fn.NewAt(x, i, j), x)
}

// Add returns a new operator node as a result of the fn.Add function.
// The first node may be null. This help to keep the code as concise as possible e.g. during accumulation.
func (g *Graph) Add(x1 Node, x2 Node) Node {
	if x1 != nil {
		return g.NewOperator(fn.NewAdd(x1, x2), x1, x2)
	}
	fake := g.NewVariable(nil, false)
	return g.NewOperator(fn.NewAdd(fake, x2), fake, x2)
}

// Sub returns a new operator node as a result of the fn.Sub function.
func (g *Graph) Sub(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewSub(x1, x2), x1, x2)
}

// SubScalar returns a new operator node as a result of the fn.SubScalar function.
func (g *Graph) SubScalar(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewSubScalar(x1, x2), x1, x2)
}

// AddScalar returns a new operator node as a result of the fn.AddScalar function.
func (g *Graph) AddScalar(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewAddScalar(x1, x2), x1, x2)
}

// ReverseSub returns a new operator node as a result of the fn.ReverseSub function.
func (g *Graph) ReverseSub(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewReverseSubScalar(x1, x2), x1, x2)
}

// Prod returns a new operator node as a result of the fn.Prod function.
func (g *Graph) Prod(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewProd(x1, x2), x1, x2)
}

// Div returns a new operator node as a result of the fn.Div function.
func (g *Graph) Div(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewDiv(x1, x2), x1, x2)
}

// ProdScalar returns a new operator node as a result of the fn.ProdScalar function.
func (g *Graph) ProdScalar(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewProdScalar(x1, x2), x1, x2)
}

// DivScalar returns a new operator node as a result of the fn.DivScalar function.
func (g *Graph) DivScalar(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewDivScalar(x1, x2), x1, x2)
}

// Mul returns a new operator node as a result of the fn.Mul function.
func (g *Graph) Mul(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewMul(x1, x2), x1, x2)
}

// Dot returns a new operator node as a result of the fn.Dot function.
func (g *Graph) Dot(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewDot(x1, x2), x1, x2)
}

// Max returns a new operator node as a result of the fn.Max function.
func (g *Graph) Max(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewMax(x1, x2), x1, x2)
}

// Min returns a new operator node as a result of the fn.Min function.
func (g *Graph) Min(x1 Node, x2 Node) Node {
	return g.NewOperator(fn.NewMin(x1, x2), x1, x2)
}

// Reshape returns a new operator node as a result of the fn.Reshape function.
func (g *Graph) Reshape(x Node, rows, columns int) Node {
	return g.NewOperator(fn.NewReshape(x, rows, columns), x)
}

// MaxPooling returns a new operator node as a result of the fn.MaxPooling function.
func (g *Graph) MaxPooling(x Node, rows, columns int) Node {
	return g.NewOperator(fn.NewMaxPooling(x, rows, columns), x)
}

// View returns a new operator node as a result of the fn.View function.
func (g *Graph) View(x Node, row, column, xStride, yStride int) Node {
	return g.NewOperator(fn.NewView(x, row, column, xStride, yStride), x)
}

// RowView returns a new operator node as a result of the fn.RowView function.
func (g *Graph) RowView(x Node, row int) Node {
	return g.NewOperator(fn.NewRowView(x, row), x)
}

// ColView returns a new operator node as a result of the fn.ColView function.
func (g *Graph) ColView(x Node, column int) Node {
	return g.NewOperator(fn.NewColView(x, column), x)
}

// Vec returns a new operator node as a result of the fn.Vec function.
func (g *Graph) Vec(x Node) Node {
	return g.NewOperator(fn.NewVec(x), x)
}

// T returns a new operator node as a result of the fn.T function.
func (g *Graph) T(x Node) Node {
	return g.NewOperator(fn.NewTranspose(x), x)
}

// Square returns a new operator node as a result of the fn.Prod(x, x) function.
func (g *Graph) Square(x Node) Node {
	return g.NewOperator(fn.NewSquare(x), x)
}

// Pow returns a new operator node as a result of the fn.Pow function.
func (g *Graph) Pow(x Node, power float64) Node {
	return g.NewOperator(fn.NewPow(x, power), x)
}

// Sqrt returns a new operator node as a result of the `Sqrt` function.
func (g *Graph) Sqrt(x Node) Node {
	return g.NewOperator(fn.NewSqrt(x), x)
}

// Tan returns a new operator node as a result of the `Tan` function.
func (g *Graph) Tan(x Node) Node {
	return g.NewOperator(fn.NewTan(x), x)
}

// Tanh returns a new operator node as a result of the `Tanh` function.
func (g *Graph) Tanh(x Node) Node {
	return g.NewOperator(fn.NewTanh(x), x)
}

// Sigmoid returns a new operator node as a result of the `Sigmoid` function.
func (g *Graph) Sigmoid(x Node) Node {
	return g.NewOperator(fn.NewSigmoid(x), x)
}

// HardSigmoid returns a new operator node as a result of the `HardSigmoid` function.
func (g *Graph) HardSigmoid(x Node) Node {
	return g.NewOperator(fn.NewHardSigmoid(x), x)
}

// HardTanh returns a new operator node as a result of the `HardTanh` function.
func (g *Graph) HardTanh(x Node) Node {
	return g.NewOperator(fn.NewHardTanh(x), x)
}

// Softsign returns a new operator node as a result of the `SoftSign` function.
func (g *Graph) Softsign(x Node) Node {
	return g.NewOperator(fn.NewSoftsign(x), x)
}

// ReLU returns a new operator node as a result of the `ReLU` function.
func (g *Graph) ReLU(x Node) Node {
	return g.NewOperator(fn.NewReLU(x), x)
}

// CeLU returns a new operator node as a result of the fn.CeLU function.
func (g *Graph) CeLU(x Node, alpha Node) Node {
	return g.NewOperator(fn.NewCeLU(x, alpha), x, alpha)
}

// GeLU returns a new operator node as a result of the fn.GeLU function.
func (g *Graph) GeLU(x Node) Node {
	return g.NewOperator(fn.NewGeLU(x), x)
}

// ELU returns a new operator node as a result of the fn.ELU function.
func (g *Graph) ELU(x Node, alpha Node) Node {
	return g.NewOperator(fn.NewELU(x, alpha), x, alpha)
}

// Swish returns a new operator node as a result of the fn.Swish function.
func (g *Graph) Swish(x Node, beta Node) Node {
	return g.NewOperator(fn.NewSwish(x, beta), x, beta)
}

// Mish returns a new operator node as a result of the `Mish` function.
func (g *Graph) Mish(x Node) Node {
	return g.NewOperator(fn.NewMish(x), x)
}

// LeakyReLU returns a new operator node as a result of the fn.LeakyReLU function.
func (g *Graph) LeakyReLU(x Node, alpha Node) Node {
	return g.NewOperator(fn.NewLeakyReLU(x, alpha), x, alpha)
}

// SeLU returns a new operator node as a result of the fn.SeLU function.
func (g *Graph) SeLU(x Node, alpha Node, scale Node) Node {
	return g.NewOperator(fn.NewSeLU(x, alpha, scale), x, alpha, scale)
}

// SoftPlus returns a new operator node as a result of the fn.SoftPlus function.
func (g *Graph) SoftPlus(x Node, beta Node, threshold Node) Node {
	return g.NewOperator(fn.NewSoftPlus(x, beta, threshold), x, beta, threshold)
}

// SoftShrink returns a new operator node as a result of the fn.SoftShrink function.
func (g *Graph) SoftShrink(x Node, lambda Node) Node {
	return g.NewOperator(fn.NewSoftShrink(x, lambda), x, lambda)
}

// Threshold returns a new operator node as a result of the fn.Threshold function.
func (g *Graph) Threshold(x Node, threshold Node, k Node) Node {
	return g.NewOperator(fn.NewThreshold(x, threshold, k), x, threshold, k)
}

// Softmax returns a new operator node as a result of the fn.Softmax function.
func (g *Graph) Softmax(x Node) Node {
	return g.NewOperator(fn.NewSoftmax(x), x)
}

// SparseMax returns a new operator node as a result of the fn.SparseMax function.
func (g *Graph) SparseMax(x Node) Node {
	return g.NewOperator(fn.NewSparseMax(x), x)
}

// SparseMaxLoss returns a new operator node as a result of the fn.SparseMaxLoss function.
func (g *Graph) SparseMaxLoss(x Node) Node {
	return g.NewOperator(fn.NewSparseMaxLoss(x), x)
}

// Sin returns a new operator node as a result of the `Sin` function.
func (g *Graph) Sin(x Node) Node {
	return g.NewOperator(fn.NewSin(x), x)
}

// Cos returns a new operator node as a result of the `Cos` function.
func (g *Graph) Cos(x Node) Node {
	return g.NewOperator(fn.NewCos(x), x)
}

// Exp returns a new operator node as a result of the `Exp` function.
func (g *Graph) Exp(x Node) Node {
	return g.NewOperator(fn.NewExp(x), x)
}

// Log returns a new operator node as a result of the `Log` function.
func (g *Graph) Log(x Node) Node {
	return g.NewOperator(fn.NewLog(x), x)
}

// Abs returns a new operator node as a result of the `Abs` function.
func (g *Graph) Abs(x Node) Node {
	return g.NewOperator(fn.NewAbs(x), x)
}

// Neg returns a new operator node as a result of the `Neg` function.
func (g *Graph) Neg(x Node) Node {
	return g.NewOperator(fn.NewNeg(x), x)
}

// Reciprocal returns a new operator node as a result of the `Reciprocal` function.
func (g *Graph) Reciprocal(x Node) Node {
	return g.NewOperator(fn.NewReciprocal(x), x)
}

// ReduceSum returns a new operator node as a result of the fn.ReduceSum function.
func (g *Graph) ReduceSum(x Node) Node {
	return g.NewOperator(fn.NewReduceSum(x), x)
}

// ReduceMean returns a new operator node as a result of the fn.ReduceMean function.
func (g *Graph) ReduceMean(x Node) Node {
	return g.NewOperator(fn.NewReduceMean(x), x)
}

// Concat returns a new operator node as a result of the fn.Concat function.
func (g *Graph) Concat(xs ...Node) Node {
	return g.NewOperator(fn.NewConcat(Operands(xs)), xs...)
}

// Stack returns a new operator node as a result of the fn.Stack function.
func (g *Graph) Stack(xs ...Node) Node {
	return g.NewOperator(fn.NewStack(Operands(xs)), xs...)
}
