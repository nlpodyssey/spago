// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/ag/fn"
	"github.com/nlpodyssey/spago/mat"
)

// Identity returns a new operator node as a result of the fn.Identity function.
func Identity[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewIdentity[T](x), x)
}

// Dropout returns a new operator node as a result of the fn.Dropout function.
// It does not apply the dropout during inference regardless the probability.
func Dropout[T mat.DType](x Node[T], p T) Node[T] {
	g := resolveGraph[T](x)
	if g.mode == Inference {
		return x
	}
	return g.NewOperator(fn.NewDropout[T](x, p, g.randGen), x)
}

// AtVec returns a new operator node as a result of the fn.AtVec function.
func AtVec[T mat.DType](x Node[T], i int) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewAtVec[T](x, i), x)
}

// At returns a new operator node as a result of the fn.At function.
func At[T mat.DType](x Node[T], i int, j int) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewAt[T](x, i, j), x)
}

// Add returns a new operator node as a result of the fn.Add function.
// As special case, the first node may be null.
// This help to keep the code as concise as possible e.g. during accumulation.
func Add[T mat.DType](x1 Node[T], x2 Node[T]) Node[T] {
	g := resolveGraph[T](x2)
	if x1 != nil {
		return g.NewOperator(fn.NewAdd[T](x1, x2), x1, x2)
	}
	placeholder := g.NewVariable(nil, false)
	return g.NewOperator(fn.NewAdd[T](placeholder, x2), placeholder, x2)
}

// Sub returns a new operator node as a result of the fn.Sub function.
func Sub[T mat.DType](x1, x2 Node[T]) Node[T] {
	return resolveGraph[T](x1).NewOperator(fn.NewSub[T](x1, x2), x1, x2)
}

// SubScalar returns a new operator node as a result of the fn.SubScalar function.
func SubScalar[T mat.DType](x1, x2 Node[T]) Node[T] {
	return resolveGraph[T](x1).NewOperator(fn.NewSubScalar[T](x1, x2), x1, x2)
}

// AddScalar returns a new operator node as a result of the fn.AddScalar function.
func AddScalar[T mat.DType](x1, x2 Node[T]) Node[T] {
	return resolveGraph[T](x1).NewOperator(fn.NewAddScalar[T](x1, x2), x1, x2)
}

// ReverseSub returns a new operator node as a result of the fn.ReverseSub function.
func ReverseSub[T mat.DType](x1, x2 Node[T]) Node[T] {
	return resolveGraph[T](x1).NewOperator(fn.NewReverseSubScalar[T](x1, x2), x1, x2)
}

// Prod returns a new operator node as a result of the fn.Prod function.
func Prod[T mat.DType](x1, x2 Node[T]) Node[T] {
	return resolveGraph[T](x1).NewOperator(fn.NewProd[T](x1, x2), x1, x2)
}

// Div returns a new operator node as a result of the fn.Div function.
func Div[T mat.DType](x1, x2 Node[T]) Node[T] {
	return resolveGraph[T](x1).NewOperator(fn.NewDiv[T](x1, x2), x1, x2)
}

// ProdScalar returns a new operator node as a result of the fn.ProdScalar function.
func ProdScalar[T mat.DType](x1, x2 Node[T]) Node[T] {
	return resolveGraph[T](x1).NewOperator(fn.NewProdScalar[T](x1, x2), x1, x2)
}

// DivScalar returns a new operator node as a result of the fn.DivScalar function.
func DivScalar[T mat.DType](x1, x2 Node[T]) Node[T] {
	return resolveGraph[T](x1).NewOperator(fn.NewDivScalar[T](x1, x2), x1, x2)
}

// Mul returns a new operator node as a result of the fn.Mul function.
func Mul[T mat.DType](x1, x2 Node[T]) Node[T] {
	return resolveGraph[T](x1).NewOperator(fn.NewMul[T](x1, x2), x1, x2)
}

// Dot returns a new operator node as a result of the fn.Dot function.
func Dot[T mat.DType](x1, x2 Node[T]) Node[T] {
	return resolveGraph[T](x1).NewOperator(fn.NewDot[T](x1, x2), x1, x2)
}

// Max returns a new operator node as a result of the fn.Max function.
func Max[T mat.DType](x1, x2 Node[T]) Node[T] {
	return resolveGraph[T](x1).NewOperator(fn.NewMax[T](x1, x2), x1, x2)
}

// Min returns a new operator node as a result of the fn.Min function.
func Min[T mat.DType](x1, x2 Node[T]) Node[T] {
	return resolveGraph[T](x1).NewOperator(fn.NewMin[T](x1, x2), x1, x2)
}

// Reshape returns a new operator node as a result of the fn.Reshape function.
func Reshape[T mat.DType](x Node[T], rows, columns int) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewReshape[T](x, rows, columns), x)
}

// MaxPooling returns a new operator node as a result of the fn.MaxPooling function.
func MaxPooling[T mat.DType](x Node[T], rows, columns int) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewMaxPooling[T](x, rows, columns), x)
}

// View returns a new operator node as a result of the fn.View function.
func View[T mat.DType](x Node[T], row, column, xStride, yStride int) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewView[T](x, row, column, xStride, yStride), x)
}

// RowView returns a new operator node as a result of the fn.RowView function.
func RowView[T mat.DType](x Node[T], row int) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewRowView[T](x, row), x)
}

// RotateR performs the right circular shift.
// `i` is the number of places by which the elements are shifted.
func RotateR[T mat.DType](x Node[T], i int) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewRotateR[T](x, i), x)
}

// ColView returns a new operator node as a result of the fn.ColView function.
func ColView[T mat.DType](x Node[T], column int) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewColView[T](x, column), x)
}

// Flatten returns a new operator node as a result of the fn.Flatten function.
func Flatten[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewFlatten[T](x), x)
}

// T returns a new operator node as a result of the fn.T function.
func T[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewTranspose[T](x), x)
}

// Square returns a new operator node as a result of the fn.Prod(x, x) function.
func Square[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewSquare[T](x), x)
}

// Pow returns a new operator node as a result of the fn.Pow function.
func Pow[T mat.DType](x Node[T], power T) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewPow[T](x, power), x)
}

// Sqrt returns a new operator node as a result of the `Sqrt` function.
func Sqrt[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewSqrt[T](x), x)
}

// Tan returns a new operator node as a result of the `Tan` function.
func Tan[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewTan[T](x), x)
}

// Tanh returns a new operator node as a result of the `Tanh` function.
func Tanh[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewTanh[T](x), x)
}

// Sigmoid returns a new operator node as a result of the `Sigmoid` function.
func Sigmoid[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewSigmoid[T](x), x)
}

// HardSigmoid returns a new operator node as a result of the `HardSigmoid` function.
func HardSigmoid[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewHardSigmoid[T](x), x)
}

// HardTanh returns a new operator node as a result of the `HardTanh` function.
func HardTanh[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewHardTanh[T](x), x)
}

// Softsign returns a new operator node as a result of the `SoftSign` function.
func Softsign[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewSoftsign[T](x), x)
}

// ReLU returns a new operator node as a result of the `ReLU` function.
func ReLU[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewReLU[T](x), x)
}

// CELU returns a new operator node as a result of the fn.CELU function.
func CELU[T mat.DType](x, alpha Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewCELU[T](x, alpha), x, alpha)
}

// GELU returns a new operator node as a result of the fn.GELU function.
func GELU[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewGELU[T](x), x)
}

// ELU returns a new operator node as a result of the fn.ELU function.
func ELU[T mat.DType](x, alpha Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewELU[T](x, alpha), x, alpha)
}

// SwishB returns a new operator node as a result of the fn.SwishB function.
func SwishB[T mat.DType](x, beta Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewSwishB[T](x, beta), x, beta)
}

// Swish returns a new operator node as a result of the fn.Swish function.
func Swish[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewSwish[T](x), x)
}

// SiLU returns a new operator node as a result of the fn.SiLU function.
func SiLU[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewSiLU[T](x), x)
}

// Mish returns a new operator node as a result of the `Mish` function.
func Mish[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewMish[T](x), x)
}

// LeakyReLU returns a new operator node as a result of the fn.LeakyReLU function.
func LeakyReLU[T mat.DType](x, alpha Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewLeakyReLU[T](x, alpha), x, alpha)
}

// SELU returns a new operator node as a result of the fn.SELU function.
func SELU[T mat.DType](x, alpha Node[T], scale Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewSELU[T](x, alpha, scale), x, alpha, scale)
}

// SoftPlus returns a new operator node as a result of the fn.SoftPlus function.
func SoftPlus[T mat.DType](x, beta, threshold Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewSoftPlus[T](x, beta, threshold), x, beta, threshold)
}

// SoftShrink returns a new operator node as a result of the fn.SoftShrink function.
func SoftShrink[T mat.DType](x, lambda Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewSoftShrink[T](x, lambda), x, lambda)
}

// Threshold returns a new operator node as a result of the fn.Threshold function.
func Threshold[T mat.DType](x, threshold, k Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewThreshold[T](x, threshold, k), x, threshold, k)
}

// Softmax returns a new operator node as a result of the fn.Softmax function.
func Softmax[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewSoftmax[T](x), x)
}

// SparseMax returns a new operator node as a result of the fn.SparseMax function.
func SparseMax[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewSparseMax[T](x), x)
}

// SparseMaxLoss returns a new operator node as a result of the fn.SparseMaxLoss function.
func SparseMaxLoss[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewSparseMaxLoss[T](x), x)
}

// Sin returns a new operator node as a result of the `Sin` function.
func Sin[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewSin[T](x), x)
}

// Cos returns a new operator node as a result of the `Cos` function.
func Cos[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewCos[T](x), x)
}

// Exp returns a new operator node as a result of the `Exp` function.
func Exp[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewExp[T](x), x)
}

// Log returns a new operator node as a result of the `Log` function.
func Log[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewLog[T](x), x)
}

// Abs returns a new operator node as a result of the `Abs` function.
func Abs[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewAbs[T](x), x)
}

// Neg returns a new operator node as a result of the `Neg` function.
func Neg[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewNeg[T](x), x)
}

// Reciprocal returns a new operator node as a result of the `Reciprocal` function.
func Reciprocal[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewReciprocal[T](x), x)
}

// ReduceSum returns a new operator node as a result of the fn.ReduceSum function.
func ReduceSum[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewReduceSum[T](x), x)
}

// ReduceMean returns a new operator node as a result of the fn.ReduceMean function.
func ReduceMean[T mat.DType](x Node[T]) Node[T] {
	return resolveGraph[T](x).NewOperator(fn.NewReduceMean[T](x), x)
}

// Concat returns a new operator node as a result of the fn.Concat function.
func Concat[T mat.DType](xs ...Node[T]) Node[T] {
	return resolveGraph[T](xs).NewOperator(fn.NewConcat(ToOperands(xs)), xs...)
}

// Stack returns a new operator node as a result of the fn.Stack function.
func Stack[T mat.DType](xs ...Node[T]) Node[T] {
	return resolveGraph[T](xs).NewOperator(fn.NewStack(ToOperands(xs)), xs...)
}

func resolveGraph[T mat.DType](i any) *Graph[T] {
	switch i := i.(type) {
	case Node[T]:
		return i.Graph()
	case []Node[T]:
		return i[0].Graph()
	default:
		panic("ag: impossible to resolve the graph")
	}
}
