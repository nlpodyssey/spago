// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/nlpodyssey/spago/ag/fn"
	"github.com/nlpodyssey/spago/mat"
)

// OpName is the enumeration-like type used for the set of operators supported
// by spaGO.
type OpName int

const (
	// OpIdentity identifies the Graph.Identity operator.
	OpIdentity OpName = iota
	// OpDropout identifies the Graph.Dropout operator.
	OpDropout
	// OpAtVec identifies the Graph.AtVec operator.
	OpAtVec
	// OpAt identifies the Graph.At operator.
	OpAt
	// OpAdd identifies the Graph.Add operator.
	OpAdd
	// OpSub identifies the Graph.Sub operator.
	OpSub
	// OpSubScalar identifies the Graph.SubScalar operator.
	OpSubScalar
	// OpAddScalar identifies the Graph.AddScalar operator.
	OpAddScalar
	// OpReverseSub identifies the Graph.ReverseSub operator.
	OpReverseSub
	// OpProd identifies the Graph.Prod operator.
	OpProd
	// OpDiv identifies the Graph.Div operator.
	OpDiv
	// OpProdScalar identifies the Graph.ProdScalar operator.
	OpProdScalar
	// OpDivScalar identifies the Graph.DivScalar operator.
	OpDivScalar
	// OpMul identifies the Graph.Mul operator.
	OpMul
	// OpDot identifies the Graph.Dot operator.
	OpDot
	// OpReshape identifies the Graph.Reshape operator.
	OpReshape
	// OpMaxPooling identifies the Graph.MaxPooling operator.
	OpMaxPooling
	// OpView identifies the Graph.View operator.
	OpView
	// OpRowView identifies the Graph.RowView operator.
	OpRowView
	// OpColView identifies the Graph.ColView operator.
	OpColView
	// OpVec identifies the Graph.Vec operator.
	OpVec
	// OpRotateR identifies the Graph.RotateR operator.
	OpRotateR
	// OpT identifies the Graph.T operator.
	OpT
	// OpSquare identifies the Graph.Square operator.
	OpSquare
	// OpPow identifies the Graph.Pow operator.
	OpPow
	// OpSqrt identifies the Graph.Sqrt operator.
	OpSqrt
	// OpTan identifies the Graph.Tan operator.
	OpTan
	// OpTanh identifies the Graph.Tanh operator.
	OpTanh
	// OpSigmoid identifies the Graph.Sigmoid operator.
	OpSigmoid
	// OpHardSigmoid identifies the Graph.HardSigmoid operator.
	OpHardSigmoid
	// OpHardTanh identifies the Graph.HardTanh operator.
	OpHardTanh
	// OpSoftsign identifies the Graph.Softsign operator.
	OpSoftsign
	// OpReLU identifies the Graph.ReLU operator.
	OpReLU
	// OpCELU identifies the Graph.CELU operator.
	OpCELU
	// OpGELU identifies the Graph.GELU operator.
	OpGELU
	// OpELU identifies the Graph.ELU operator.
	OpELU
	// OpPositiveELU identifies the Graph.PositiveELU operator.
	OpPositiveELU
	// OpSwishB identifies the Graph.SwishB operator.
	OpSwishB
	// OpSwish identifies the Graph.Swish operator.
	OpSwish
	// OpSiLU identifies the Graph.SiLU operator.
	OpSiLU
	// OpMish identifies the Graph.Mish operator.
	OpMish
	// OpLeakyReLU identifies the Graph.LeakyReLU operator.
	OpLeakyReLU
	// OpSELU identifies the Graph.SELU operator.
	OpSELU
	// OpSoftPlus identifies the Graph.SoftPlus operator.
	OpSoftPlus
	// OpSoftShrink identifies the Graph.SoftShrink operator.
	OpSoftShrink
	// OpThreshold identifies the Graph.Threshold operator.
	OpThreshold
	// OpSoftmax identifies the Graph.Softmax operator.
	OpSoftmax
	// OpLogSoftmax identifies the Graph.LogSoftmax operator.
	OpLogSoftmax
	// OpSparseMax identifies the Graph.SparseMax operator.
	OpSparseMax
	// OpSparseMaxLoss identifies the Graph.SparseMaxLoss operator.
	OpSparseMaxLoss
	// OpSin identifies the Graph.Sin operator.
	OpSin
	// OpCos identifies the Graph.Cos operator.
	OpCos
	// OpExp identifies the Graph.Exp operator.
	OpExp
	// OpLog identifies the Graph.Log operator.
	OpLog
	// OpAbs identifies the Graph.Abs operator.
	OpAbs
	// OpNeg identifies the Graph.Neg operator.
	OpNeg
	// OpReciprocal identifies the Graph.Reciprocal operator.
	OpReciprocal
	// OpMax identifies the Graph.Max operator.
	OpMax
	// OpMin identifies the Graph.Min operator.
	OpMin
	// OpReduceSum identifies the Graph.ReduceSum operator.
	OpReduceSum
	// OpReduceMean identifies the Graph.ReduceMean operator.
	OpReduceMean
	// OpMean identifies the Graph.Mean operator.
	OpMean
	// OpSum identifies the Graph.Sum operator.
	OpSum
	// OpConcat identifies the Graph.Concat operator.
	OpConcat
	// OpStack identifies the Graph.Stack operator.
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
	OpRotateR:       "RotateR",
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
	OpCELU:          "CELU",
	OpGELU:          "GELU",
	OpELU:           "ELU",
	OpPositiveELU:   "PositiveELU",
	OpSwishB:        "SwishB",
	OpSwish:         "Swish",
	OpSiLU:          "SiLU",
	OpMish:          "Mish",
	OpLeakyReLU:     "LeakyReLU",
	OpSELU:          "SELU",
	OpSoftPlus:      "SoftPlus",
	OpSoftShrink:    "SoftShrink",
	OpThreshold:     "Threshold",
	OpSoftmax:       "Softmax",
	OpLogSoftmax:    "LogSoftmax",
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

// strToOpName is the inverse map of opNameToMethodName.
var strToOpName = func() map[string]OpName {
	invMap := make(map[string]OpName)
	for k, v := range opNameToMethodName {
		invMap[v] = k
		invMap[strings.ToLower(v)] = k
	}
	return invMap
}()

// GetOpName maps a string to an operator.
// It panics if the string does not match any operator (not even using lowercase).
func GetOpName(str string) (OpName, error) {
	if value, ok := strToOpName[str]; ok {
		return value, nil
	}
	return -1, fmt.Errorf("ag: unknown operator %s", str)
}

// Invoke returns a new node as a result of the application of the input operator.
func Invoke[T mat.DType](operator OpName, xs ...Node[T]) Node[T] {
	v := reflect.ValueOf(xs[0].Graph()).MethodByName(opNameToMethodName[operator])
	args := make([]reflect.Value, len(xs))
	for i, x := range xs {
		args[i] = reflect.ValueOf(x)
	}
	ret := v.Call(args)
	return ret[0].Interface().(Node[T])
}

// Identity returns a new operator node as a result of the fn.Identity function.
func Identity[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewIdentity[T](x), x)
}

// Dropout returns a new operator node as a result of the fn.Dropout function.
func Dropout[T mat.DType](x Node[T], p T) Node[T] {
	g := x.Graph()
	return x.Graph().NewOperator(fn.NewDropout[T](x, p, g.randGen), x)
}

// AtVec returns a new operator node as a result of the fn.AtVec function.
func AtVec[T mat.DType](x Node[T], i int) Node[T] {
	return x.Graph().NewOperator(fn.NewAtVec[T](x, i), x)
}

// At returns a new operator node as a result of the fn.At function.
func At[T mat.DType](x Node[T], i int, j int) Node[T] {
	return x.Graph().NewOperator(fn.NewAt[T](x, i, j), x)
}

// Add returns a new operator node as a result of the fn.Add function.
// As special case, the first node may be null.
// This help to keep the code as concise as possible e.g. during accumulation.
func Add[T mat.DType](x1 Node[T], x2 Node[T]) Node[T] {
	g := x2.Graph()
	if x1 != nil {
		return g.NewOperator(fn.NewAdd[T](x1, x2), x1, x2)
	}
	fake := g.NewVariable(nil, false)
	return g.NewOperator(fn.NewAdd[T](fake, x2), fake, x2)
}

// Sub returns a new operator node as a result of the fn.Sub function.
func Sub[T mat.DType](x1, x2 Node[T]) Node[T] {
	return x1.Graph().NewOperator(fn.NewSub[T](x1, x2), x1, x2)
}

// SubScalar returns a new operator node as a result of the fn.SubScalar function.
func SubScalar[T mat.DType](x1, x2 Node[T]) Node[T] {
	return x1.Graph().NewOperator(fn.NewSubScalar[T](x1, x2), x1, x2)
}

// AddScalar returns a new operator node as a result of the fn.AddScalar function.
func AddScalar[T mat.DType](x1, x2 Node[T]) Node[T] {
	return x1.Graph().NewOperator(fn.NewAddScalar[T](x1, x2), x1, x2)
}

// ReverseSub returns a new operator node as a result of the fn.ReverseSub function.
func ReverseSub[T mat.DType](x1, x2 Node[T]) Node[T] {
	return x1.Graph().NewOperator(fn.NewReverseSubScalar[T](x1, x2), x1, x2)
}

// Prod returns a new operator node as a result of the fn.Prod function.
func Prod[T mat.DType](x1, x2 Node[T]) Node[T] {
	return x1.Graph().NewOperator(fn.NewProd[T](x1, x2), x1, x2)
}

// Div returns a new operator node as a result of the fn.Div function.
func Div[T mat.DType](x1, x2 Node[T]) Node[T] {
	return x1.Graph().NewOperator(fn.NewDiv[T](x1, x2), x1, x2)
}

// ProdScalar returns a new operator node as a result of the fn.ProdScalar function.
func ProdScalar[T mat.DType](x1, x2 Node[T]) Node[T] {
	return x1.Graph().NewOperator(fn.NewProdScalar[T](x1, x2), x1, x2)
}

// DivScalar returns a new operator node as a result of the fn.DivScalar function.
func DivScalar[T mat.DType](x1, x2 Node[T]) Node[T] {
	return x1.Graph().NewOperator(fn.NewDivScalar[T](x1, x2), x1, x2)
}

// Mul returns a new operator node as a result of the fn.Mul function.
func Mul[T mat.DType](x1, x2 Node[T]) Node[T] {
	return x1.Graph().NewOperator(fn.NewMul[T](x1, x2), x1, x2)
}

// Dot returns a new operator node as a result of the fn.Dot function.
func Dot[T mat.DType](x1, x2 Node[T]) Node[T] {
	return x1.Graph().NewOperator(fn.NewDot[T](x1, x2), x1, x2)
}

// Max returns a new operator node as a result of the fn.Max function.
func Max[T mat.DType](x1, x2 Node[T]) Node[T] {
	return x1.Graph().NewOperator(fn.NewMax[T](x1, x2), x1, x2)
}

// Min returns a new operator node as a result of the fn.Min function.
func Min[T mat.DType](x1, x2 Node[T]) Node[T] {
	return x1.Graph().NewOperator(fn.NewMin[T](x1, x2), x1, x2)
}

// Reshape returns a new operator node as a result of the fn.Reshape function.
func Reshape[T mat.DType](x Node[T], rows, columns int) Node[T] {
	return x.Graph().NewOperator(fn.NewReshape[T](x, rows, columns), x)
}

// MaxPooling returns a new operator node as a result of the fn.MaxPooling function.
func MaxPooling[T mat.DType](x Node[T], rows, columns int) Node[T] {
	return x.Graph().NewOperator(fn.NewMaxPooling[T](x, rows, columns), x)
}

// View returns a new operator node as a result of the fn.View function.
func View[T mat.DType](x Node[T], row, column, xStride, yStride int) Node[T] {
	return x.Graph().NewOperator(fn.NewView[T](x, row, column, xStride, yStride), x)
}

// RowView returns a new operator node as a result of the fn.RowView function.
func RowView[T mat.DType](x Node[T], row int) Node[T] {
	return x.Graph().NewOperator(fn.NewRowView[T](x, row), x)
}

// RotateR performs the right circular shift.
// `i` is the number of places by which the elements are shifted.
func RotateR[T mat.DType](x Node[T], i int) Node[T] {
	return x.Graph().NewOperator(fn.NewRotateR[T](x, i), x)
}

// ColView returns a new operator node as a result of the fn.ColView function.
func ColView[T mat.DType](x Node[T], column int) Node[T] {
	return x.Graph().NewOperator(fn.NewColView[T](x, column), x)
}

// Vec returns a new operator node as a result of the fn.Vec function.
func Vec[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewVec[T](x), x)
}

// T returns a new operator node as a result of the fn.T function.
func T[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewTranspose[T](x), x)
}

// Square returns a new operator node as a result of the fn.Prod(x, x) function.
func Square[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewSquare[T](x), x)
}

// Pow returns a new operator node as a result of the fn.Pow function.
func Pow[T mat.DType](x Node[T], power T) Node[T] {
	return x.Graph().NewOperator(fn.NewPow[T](x, power), x)
}

// Sqrt returns a new operator node as a result of the `Sqrt` function.
func Sqrt[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewSqrt[T](x), x)
}

// Tan returns a new operator node as a result of the `Tan` function.
func Tan[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewTan[T](x), x)
}

// Tanh returns a new operator node as a result of the `Tanh` function.
func Tanh[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewTanh[T](x), x)
}

// Sigmoid returns a new operator node as a result of the `Sigmoid` function.
func Sigmoid[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewSigmoid[T](x), x)
}

// HardSigmoid returns a new operator node as a result of the `HardSigmoid` function.
func HardSigmoid[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewHardSigmoid[T](x), x)
}

// HardTanh returns a new operator node as a result of the `HardTanh` function.
func HardTanh[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewHardTanh[T](x), x)
}

// Softsign returns a new operator node as a result of the `SoftSign` function.
func Softsign[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewSoftsign[T](x), x)
}

// ReLU returns a new operator node as a result of the `ReLU` function.
func ReLU[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewReLU[T](x), x)
}

// CELU returns a new operator node as a result of the fn.CELU function.
func CELU[T mat.DType](x, alpha Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewCELU[T](x, alpha), x, alpha)
}

// GELU returns a new operator node as a result of the fn.GELU function.
func GELU[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewGELU[T](x), x)
}

// ELU returns a new operator node as a result of the fn.ELU function.
func ELU[T mat.DType](x, alpha Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewELU[T](x, alpha), x, alpha)
}

// SwishB returns a new operator node as a result of the fn.SwishB function.
func SwishB[T mat.DType](x, beta Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewSwishB[T](x, beta), x, beta)
}

// Swish returns a new operator node as a result of the fn.Swish function.
func Swish[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewSwish[T](x), x)
}

// SiLU returns a new operator node as a result of the fn.SiLU function.
func SiLU[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewSiLU[T](x), x)
}

// Mish returns a new operator node as a result of the `Mish` function.
func Mish[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewMish[T](x), x)
}

// LeakyReLU returns a new operator node as a result of the fn.LeakyReLU function.
func LeakyReLU[T mat.DType](x, alpha Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewLeakyReLU[T](x, alpha), x, alpha)
}

// SELU returns a new operator node as a result of the fn.SELU function.
func SELU[T mat.DType](x, alpha Node[T], scale Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewSELU[T](x, alpha, scale), x, alpha, scale)
}

// SoftPlus returns a new operator node as a result of the fn.SoftPlus function.
func SoftPlus[T mat.DType](x, beta, threshold Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewSoftPlus[T](x, beta, threshold), x, beta, threshold)
}

// SoftShrink returns a new operator node as a result of the fn.SoftShrink function.
func SoftShrink[T mat.DType](x, lambda Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewSoftShrink[T](x, lambda), x, lambda)
}

// Threshold returns a new operator node as a result of the fn.Threshold function.
func Threshold[T mat.DType](x, threshold, k Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewThreshold[T](x, threshold, k), x, threshold, k)
}

// Softmax returns a new operator node as a result of the fn.Softmax function.
func Softmax[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewSoftmax[T](x), x)
}

// SparseMax returns a new operator node as a result of the fn.SparseMax function.
func SparseMax[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewSparseMax[T](x), x)
}

// SparseMaxLoss returns a new operator node as a result of the fn.SparseMaxLoss function.
func SparseMaxLoss[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewSparseMaxLoss[T](x), x)
}

// Sin returns a new operator node as a result of the `Sin` function.
func Sin[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewSin[T](x), x)
}

// Cos returns a new operator node as a result of the `Cos` function.
func Cos[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewCos[T](x), x)
}

// Exp returns a new operator node as a result of the `Exp` function.
func Exp[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewExp[T](x), x)
}

// Log returns a new operator node as a result of the `Log` function.
func Log[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewLog[T](x), x)
}

// Abs returns a new operator node as a result of the `Abs` function.
func Abs[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewAbs[T](x), x)
}

// Neg returns a new operator node as a result of the `Neg` function.
func Neg[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewNeg[T](x), x)
}

// Reciprocal returns a new operator node as a result of the `Reciprocal` function.
func Reciprocal[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewReciprocal[T](x), x)
}

// ReduceSum returns a new operator node as a result of the fn.ReduceSum function.
func ReduceSum[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewReduceSum[T](x), x)
}

// ReduceMean returns a new operator node as a result of the fn.ReduceMean function.
func ReduceMean[T mat.DType](x Node[T]) Node[T] {
	return x.Graph().NewOperator(fn.NewReduceMean[T](x), x)
}

// Concat returns a new operator node as a result of the fn.Concat function.
func Concat[T mat.DType](xs ...Node[T]) Node[T] {
	return xs[0].Graph().NewOperator(fn.NewConcat(ToOperands(xs)), xs...)
}

// Stack returns a new operator node as a result of the fn.Stack function.
func Stack[T mat.DType](xs ...Node[T]) Node[T] {
	return xs[0].Graph().NewOperator(fn.NewStack(ToOperands(xs)), xs...)
}
