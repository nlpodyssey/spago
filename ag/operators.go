// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/ag/fn"
	"github.com/nlpodyssey/spago/mat"
)

// Abs returns a new operator node as a result of the `Abs` function.
func Abs[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewAbs[T](x))
}

// Add returns a new operator node as a result of the fn.Add function.
// As special case, the first node may be null.
// This help to keep the code as concise as possible e.g. during accumulation.
func Add[T mat.DType](x1 Node[T], x2 Node[T]) Node[T] {
	if x1 != nil {
		return NewOperator[T](fn.NewAdd[T](x1, x2))
	}
	placeholder := NewVariable[T](nil, false)
	return NewOperator[T](fn.NewAdd[T](placeholder, x2))
}

// AddScalar returns a new operator node as a result of the fn.AddScalar function.
func AddScalar[T mat.DType](x1, x2 Node[T]) Node[T] {
	return NewOperator[T](fn.NewAddScalar[T](x1, x2))
}

// AppendRows returns a new operator node as a result of the fn.AppendRows function.
func AppendRows[T mat.DType](x Node[T], vs ...Node[T]) Node[T] {
	return NewOperator[T](fn.NewAppendRows[T](x, vs...))
}

// At returns a new operator node as a result of the fn.At function.
func At[T mat.DType](x Node[T], i int, j int) Node[T] {
	return NewOperator[T](fn.NewAt[T](x, i, j))
}

// AtVec returns a new operator node as a result of the fn.AtVec function.
func AtVec[T mat.DType](x Node[T], i int) Node[T] {
	return NewOperator[T](fn.NewAtVec[T](x, i))
}

// CELU returns a new operator node as a result of the fn.CELU function.
func CELU[T mat.DType](x, alpha Node[T]) Node[T] {
	return NewOperator[T](fn.NewCELU[T](x, alpha))
}

// ColView returns a new operator node as a result of the fn.ColView function.
func ColView[T mat.DType](x Node[T], column int) Node[T] {
	return NewOperator[T](fn.NewColView[T](x, column))
}

// Concat returns a new operator node as a result of the fn.Concat function.
func Concat[T mat.DType](xs ...Node[T]) Node[T] {
	return NewOperator[T](fn.NewConcat[T](xs))
}

// Cos returns a new operator node as a result of the `Cos` function.
func Cos[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewCos[T](x))
}

// Div returns a new operator node as a result of the fn.Div function.
func Div[T mat.DType](x1, x2 Node[T]) Node[T] {
	return NewOperator[T](fn.NewDiv[T](x1, x2))
}

// DivScalar returns a new operator node as a result of the fn.DivScalar function.
func DivScalar[T mat.DType](x1, x2 Node[T]) Node[T] {
	return NewOperator[T](fn.NewDivScalar[T](x1, x2))
}

// Dot returns a new operator node as a result of the fn.Dot function.
func Dot[T mat.DType](x1, x2 Node[T]) Node[T] {
	return NewOperator[T](fn.NewDot[T](x1, x2))
}

// Dropout returns a new operator node as a result of the fn.Dropout function.
// It does not apply the dropout during inference regardless the probability.
func Dropout[T mat.DType](x Node[T], p T) Node[T] {
	if p == 0.0 {
		return x
	}
	return NewOperator[T](fn.NewDropout[T](x, p, globalGenerator[T]()))
}

// ELU returns a new operator node as a result of the fn.ELU function.
func ELU[T mat.DType](x, alpha Node[T]) Node[T] {
	return NewOperator[T](fn.NewELU[T](x, alpha))
}

// Exp returns a new operator node as a result of the `Exp` function.
func Exp[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewExp[T](x))
}

// Flatten returns a new operator node as a result of the fn.Flatten function.
func Flatten[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewFlatten[T](x))
}

// GELU returns a new operator node as a result of the fn.GELU function.
func GELU[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewGELU[T](x))
}

// HardSigmoid returns a new operator node as a result of the `HardSigmoid` function.
func HardSigmoid[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewHardSigmoid[T](x))
}

// HardTanh returns a new operator node as a result of the `HardTanh` function.
func HardTanh[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewHardTanh[T](x))
}

// Identity returns a new operator node as a result of the fn.Identity function.
func Identity[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewIdentity[T](x))
}

// LeakyReLU returns a new operator node as a result of the fn.LeakyReLU function.
func LeakyReLU[T mat.DType](x, alpha Node[T]) Node[T] {
	return NewOperator[T](fn.NewLeakyReLU[T](x, alpha))
}

// Log returns a new operator node as a result of the `Log` function.
func Log[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewLog[T](x))
}

// Max returns a new operator node as a result of the fn.Max function.
func Max[T mat.DType](x1, x2 Node[T]) Node[T] {
	return NewOperator[T](fn.NewMax[T](x1, x2))
}

// MaxPooling returns a new operator node as a result of the fn.MaxPooling function.
func MaxPooling[T mat.DType](x Node[T], rows, columns int) Node[T] {
	return NewOperator[T](fn.NewMaxPooling[T](x, rows, columns))
}

// Min returns a new operator node as a result of the fn.Min function.
func Min[T mat.DType](x1, x2 Node[T]) Node[T] {
	return NewOperator[T](fn.NewMin[T](x1, x2))
}

// Mish returns a new operator node as a result of the `Mish` function.
func Mish[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewMish[T](x))
}

// Mul returns a new operator node as a result of the fn.Mul function.
func Mul[T mat.DType](x1, x2 Node[T]) Node[T] {
	return NewOperator[T](fn.NewMul[T](x1, x2))
}

func MulT[T mat.DType](x1, x2 Node[T]) Node[T] {
	return NewOperator[T](fn.NewMulT[T](x1, x2))
}

// Neg returns a new operator node as a result of the `Neg` function.
func Neg[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewNeg[T](x))
}

// Pow returns a new operator node as a result of the fn.Pow function.
func Pow[T mat.DType](x Node[T], power T) Node[T] {
	return NewOperator[T](fn.NewPow[T](x, power))
}

// Prod returns a new operator node as a result of the fn.Prod function.
func Prod[T mat.DType](x1, x2 Node[T]) Node[T] {
	return NewOperator[T](fn.NewProd[T](x1, x2))
}

// ProdScalar returns a new operator node as a result of the fn.ProdScalar function.
func ProdScalar[T mat.DType](x1, x2 Node[T]) Node[T] {
	return NewOperator[T](fn.NewProdScalar[T](x1, x2))
}

// Reciprocal returns a new operator node as a result of the `Reciprocal` function.
func Reciprocal[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewReciprocal[T](x))
}

// ReduceMax returns a new operator node as a result of the fn.ReduceMax function.
func ReduceMax[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewReduceMax[T](x))
}

// ReduceMean returns a new operator node as a result of the fn.ReduceMean function.
func ReduceMean[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewReduceMean[T](x))
}

// ReduceSum returns a new operator node as a result of the fn.ReduceSum function.
func ReduceSum[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewReduceSum[T](x))
}

// ReLU returns a new operator node as a result of the `ReLU` function.
func ReLU[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewReLU[T](x))
}

// Reshape returns a new operator node as a result of the fn.Reshape function.
func Reshape[T mat.DType](x Node[T], rows, columns int) Node[T] {
	return NewOperator[T](fn.NewReshape[T](x, rows, columns))
}

// ReverseSub returns a new operator node as a result of the fn.ReverseSub function.
func ReverseSub[T mat.DType](x1, x2 Node[T]) Node[T] {
	return NewOperator[T](fn.NewReverseSubScalar[T](x1, x2))
}

// RotateR performs the right circular shift.
// `i` is the number of places by which the elements are shifted.
func RotateR[T mat.DType](x Node[T], i int) Node[T] {
	return NewOperator[T](fn.NewRotateR[T](x, i))
}

// RowView returns a new operator node as a result of the fn.RowView function.
func RowView[T mat.DType](x Node[T], row int) Node[T] {
	return NewOperator[T](fn.NewRowView[T](x, row))
}

// ScalarMax returns a new operator node as a result of the fn.ScalarMax function.
func ScalarMax[T mat.DType](xs []Node[T]) Node[T] {
	return NewOperator[T](fn.NewScalarMax[T](xs))
}

// SELU returns a new operator node as a result of the fn.SELU function.
func SELU[T mat.DType](x, alpha Node[T], scale Node[T]) Node[T] {
	return NewOperator[T](fn.NewSELU[T](x, alpha, scale))
}

// Sigmoid returns a new operator node as a result of the `Sigmoid` function.
func Sigmoid[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewSigmoid[T](x))
}

// SiLU returns a new operator node as a result of the fn.SiLU function.
func SiLU[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewSiLU[T](x))
}

// Sin returns a new operator node as a result of the `Sin` function.
func Sin[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewSin[T](x))
}

// Slice returns a new operator node as a result of the fn.Slice function.
func Slice[T mat.DType](x Node[T], fromRow, fromCol, toRow, toCol int) Node[T] {
	return NewOperator[T](fn.NewSlice[T](x, fromRow, fromCol, toRow, toCol))
}

// Softmax returns a new operator node as a result of the fn.Softmax function.
func Softmax[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewSoftmax[T](x))
}

// SoftPlus returns a new operator node as a result of the fn.SoftPlus function.
func SoftPlus[T mat.DType](x, beta, threshold Node[T]) Node[T] {
	return NewOperator[T](fn.NewSoftPlus[T](x, beta, threshold))
}

// SoftShrink returns a new operator node as a result of the fn.SoftShrink function.
func SoftShrink[T mat.DType](x, lambda Node[T]) Node[T] {
	return NewOperator[T](fn.NewSoftShrink[T](x, lambda))
}

// Softsign returns a new operator node as a result of the `SoftSign` function.
func Softsign[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewSoftsign[T](x))
}

// SparseMax returns a new operator node as a result of the fn.SparseMax function.
func SparseMax[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewSparseMax[T](x))
}

// SparseMaxLoss returns a new operator node as a result of the fn.SparseMaxLoss function.
func SparseMaxLoss[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewSparseMaxLoss[T](x))
}

// Sqrt returns a new operator node as a result of the `Sqrt` function.
func Sqrt[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewSqrt[T](x))
}

// Square returns a new operator node as a result of the fn.Prod(x, x) function.
func Square[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewSquare[T](x))
}

// Stack returns a new operator node as a result of the fn.Stack function.
func Stack[T mat.DType](xs ...Node[T]) Node[T] {
	return NewOperator[T](fn.NewStack[T](xs))
}

// Sub returns a new operator node as a result of the fn.Sub function.
func Sub[T mat.DType](x1, x2 Node[T]) Node[T] {
	return NewOperator[T](fn.NewSub[T](x1, x2))
}

// SubScalar returns a new operator node as a result of the fn.SubScalar function.
func SubScalar[T mat.DType](x1, x2 Node[T]) Node[T] {
	return NewOperator[T](fn.NewSubScalar[T](x1, x2))
}

// Swish returns a new operator node as a result of the fn.Swish function.
func Swish[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewSwish[T](x))
}

// SwishB returns a new operator node as a result of the fn.SwishB function.
func SwishB[T mat.DType](x, beta Node[T]) Node[T] {
	return NewOperator[T](fn.NewSwishB[T](x, beta))
}

// T returns a new operator node as a result of the fn.T function.
func T[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewTranspose[T](x))
}

// Tan returns a new operator node as a result of the `Tan` function.
func Tan[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewTan[T](x))
}

// Tanh returns a new operator node as a result of the `Tanh` function.
func Tanh[T mat.DType](x Node[T]) Node[T] {
	return NewOperator[T](fn.NewTanh[T](x))
}

// Threshold returns a new operator node as a result of the fn.Threshold function.
func Threshold[T mat.DType](x, threshold, k Node[T]) Node[T] {
	return NewOperator[T](fn.NewThreshold[T](x, threshold, k))
}
