// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"fmt"
	"math"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/gradfn"
)

// Abs returns a new operator node as a result of the `Abs` function.
func Abs(x DualValue) DualValue {
	return NewOperator(gradfn.NewAbs(x)).Run()
}

// Add returns a new operator node as a result of the gradfn.Add function.
// As special case, the first node may be null.
// This help to keep the code as concise as possible e.g. during accumulation.
func Add(x1 DualValue, x2 DualValue) DualValue {
	if x1 == nil {
		return Copy(x2) // return a copy of `x2` as is
	}
	return NewOperator(gradfn.NewAdd(x1, x2)).Run(true)
}

// AddScalar returns a new operator node as a result of the gradfn.AddScalar function.
func AddScalar(x1, x2 DualValue) DualValue {
	return NewOperator(gradfn.NewAddScalar(x1, x2)).Run()
}

// Affine returns a new operator node as a result of the gradfn.Affine function.
func Affine(b, w1, x1 DualValue, wxPairs ...DualValue) DualValue {
	return NewOperator(gradfn.NewAffine(b, w1, x1, wxPairs...)).Run(true)
}

// AppendRows returns a new operator node as a result of the gradfn.AppendRows function.
func AppendRows(x DualValue, vs ...DualValue) DualValue {
	return NewOperator(gradfn.NewAppendRows(x, vs...)).Run()
}

// At returns a new operator node as a result of the gradfn.At function.
func At(x DualValue, indices ...int) DualValue {
	return NewOperator(gradfn.NewAt(x, indices...)).Run()
}

// CELU returns a new operator node as a result of the gradfn.CELU function.
func CELU(x, alpha DualValue) DualValue {
	return NewOperator(gradfn.NewCELU(x, alpha)).Run()
}

// ColView returns a new operator node as a result of the gradfn.ColView function.
func ColView(x DualValue, column int) DualValue {
	return NewOperator(gradfn.NewColView(x, column)).Run()
}

// Concat returns a new operator node as a result of the gradfn.Concat function.
func Concat(xs ...DualValue) DualValue {
	return NewOperator(gradfn.NewConcat(xs)).Run()
}

// Cos returns a new operator node as a result of the `Cos` function.
func Cos(x DualValue) DualValue {
	return NewOperator(gradfn.NewCos(x)).Run()
}

// Div returns a new operator node as a result of the gradfn.Div function.
func Div(x1, x2 DualValue) DualValue {
	return NewOperator(gradfn.NewDiv(x1, x2)).Run()
}

// DivScalar returns a new operator node as a result of the gradfn.DivScalar function.
func DivScalar(x1, x2 DualValue) DualValue {
	return NewOperator(gradfn.NewDivScalar(x1, x2)).Run()
}

// Dot returns a new operator node as a result of the gradfn.Dot function.
func Dot(x1, x2 DualValue) DualValue {
	return NewOperator(gradfn.NewDot(x1, x2)).Run()
}

// DropoutFunc returns a function to create a Dropout operator working with the given dropout probability.
func DropoutFunc(p float64) func(x DualValue) DualValue {
	return func(x DualValue) DualValue {
		if p == 0.0 {
			return x
		}
		return NewOperator(gradfn.NewDropout(x, p, globalGenerator)).Run()
	}
}

// Dropout returns a new operator node as a result of the gradfn.Dropout function.
// If the dropout probability is zero, the operator will not be created,
// so the input itself is returned directly.
func Dropout(x DualValue, p float64) DualValue {
	if p == 0.0 {
		return x
	}
	return NewOperator(gradfn.NewDropout(x, p, globalGenerator)).Run()
}

// ELU returns a new operator node as a result of the gradfn.ELU function.
func ELU(x, alpha DualValue) DualValue {
	return NewOperator(gradfn.NewELU(x, alpha)).Run()
}

// Exp returns a new operator node as a result of the `Exp` function.
func Exp(x DualValue) DualValue {
	return NewOperator(gradfn.NewExp(x)).Run()
}

// Flatten returns a new operator node as a result of the gradfn.Flatten function.
func Flatten(x DualValue) DualValue {
	return NewOperator(gradfn.NewFlatten(x)).Run()
}

// GELU returns a new operator node as a result of the gradfn.GELU function.
func GELU(x DualValue) DualValue {
	return NewOperator(gradfn.NewGELU(x)).Run()
}

// HardSigmoid returns a new operator node as a result of the `HardSigmoid` function.
func HardSigmoid(x DualValue) DualValue {
	return NewOperator(gradfn.NewHardSigmoid(x)).Run()
}

// HardTanh returns a new operator node as a result of the `HardTanh` function.
func HardTanh(x DualValue) DualValue {
	return NewOperator(gradfn.NewHardTanh(x)).Run()
}

// Copy returns a new operator node as a result of the gradfn.Copy function.
func Copy(x DualValue) DualValue {
	return NewOperator(gradfn.NewCopy(x)).Run()
}

// LeakyReLU returns a new operator node as a result of the gradfn.LeakyReLU function.
func LeakyReLU(x, alpha DualValue) DualValue {
	return NewOperator(gradfn.NewLeakyReLU(x, alpha)).Run()
}

// Log returns a new operator node as a result of the `Log` function.
func Log(x DualValue) DualValue {
	return NewOperator(gradfn.NewLog(x)).Run()
}

// Max returns a new operator node as a result of the gradfn.Max function.
func Max(x1, x2 DualValue) DualValue {
	return NewOperator(gradfn.NewMax(x1, x2)).Run()
}

// MaxPooling returns a new operator node as a result of the gradfn.MaxPooling function.
func MaxPooling(x DualValue, rows, columns int) DualValue {
	return NewOperator(gradfn.NewMaxPooling(x, rows, columns)).Run()
}

// Min returns a new operator node as a result of the gradfn.Min function.
func Min(x1, x2 DualValue) DualValue {
	return NewOperator(gradfn.NewMin(x1, x2)).Run()
}

// Mish returns a new operator node as a result of the `Mish` function.
func Mish(x DualValue) DualValue {
	return NewOperator(gradfn.NewMish(x)).Run()
}

// Mul returns a new operator node as a result of the gradfn.Mul function.
func Mul(x1, x2 DualValue) DualValue {
	return NewOperator(gradfn.NewMul(x1, x2)).Run(true)
}

func MulT(x1, x2 DualValue) DualValue {
	return NewOperator(gradfn.NewMulT(x1, x2)).Run(true)
}

// Neg returns a new operator node as a result of the `Neg` function.
func Neg(x DualValue) DualValue {
	return NewOperator(gradfn.NewNeg(x)).Run()
}

// Pow returns a new operator node as a result of the gradfn.Pow function.
func Pow(x DualValue, power float64) DualValue {
	return NewOperator(gradfn.NewPow(x, power)).Run()
}

// Prod returns a new operator node as a result of the gradfn.Prod function.
func Prod(x1, x2 DualValue) DualValue {
	return NewOperator(gradfn.NewProd(x1, x2)).Run()
}

// ProdScalar returns a new operator node as a result of the gradfn.ProdScalar function.
func ProdScalar(x1, x2 DualValue) DualValue {
	return NewOperator(gradfn.NewProdScalar(x1, x2)).Run(true)
}

// Reciprocal returns a new operator node as a result of the `Reciprocal` function.
func Reciprocal(x DualValue) DualValue {
	return NewOperator(gradfn.NewReciprocal(x)).Run()
}

// ReduceMax returns a new operator node as a result of the gradfn.ReduceMax function.
func ReduceMax(x DualValue) DualValue {
	return NewOperator(gradfn.NewReduceMax(x)).Run()
}

// ReduceMean returns a new operator node as a result of the gradfn.ReduceMean function.
func ReduceMean(x DualValue) DualValue {
	return NewOperator(gradfn.NewReduceMean(x)).Run()
}

// ReduceSum returns a new operator node as a result of the gradfn.ReduceSum function.
func ReduceSum(x DualValue) DualValue {
	return NewOperator(gradfn.NewReduceSum(x)).Run()
}

// ReLU returns a new operator node as a result of the `ReLU` function.
func ReLU(x DualValue) DualValue {
	return NewOperator(gradfn.NewReLU(x)).Run(true)
}

// Reshape returns a new operator node as a result of the gradfn.Reshape function.
func Reshape(x DualValue, rows, columns int) DualValue {
	return NewOperator(gradfn.NewReshape(x, rows, columns)).Run()
}

// ReverseSub returns a new operator node as a result of the fn.ReverseSub function.
func ReverseSub(x1, x2 DualValue) DualValue {
	return NewOperator(gradfn.NewReverseSubScalar(x1, x2)).Run()
}

// ReverseSubOne returns a new operator node as a result of applying reverse subtraction with 1.0 to the input using the fn.ReverseSub function.
func ReverseSubOne(x DualValue) DualValue {
	return NewOperator(gradfn.NewReverseSubScalar(x, DualValue(mat.Scalar(1.0)))).Run()
}

// RotateR performs the right circular shift.
// `i` is the number of places by which the elements are shifted.
func RotateR(x DualValue, i int) DualValue {
	return NewOperator(gradfn.NewRotateR(x, i)).Run()
}

// RowView returns a new operator node as a result of the gradfn.RowView function.
func RowView(x DualValue, row int) DualValue {
	return NewOperator(gradfn.NewRowView(x, row)).Run()
}

// ScalarMax returns a new operator node as a result of the gradfn.ScalarMax function.
func ScalarMax(xs []DualValue) DualValue {
	return NewOperator(gradfn.NewScalarMax(xs)).Run()
}

// SELU returns a new operator node as a result of the gradfn.SELU function.
func SELU(x, alpha DualValue, scale DualValue) DualValue {
	return NewOperator(gradfn.NewSELU(x, alpha, scale)).Run()
}

// Sigmoid returns a new operator node as a result of the `Sigmoid` function.
func Sigmoid(x DualValue) DualValue {
	return NewOperator(gradfn.NewSigmoid(x)).Run()
}

// SiLU returns a new operator node as a result of the fn.SiLU function.
func SiLU(x DualValue) DualValue {
	return NewOperator(gradfn.NewSiLU(x)).Run()
}

// Sin returns a new operator node as a result of the `Sin` function.
func Sin(x DualValue) DualValue {
	return NewOperator(gradfn.NewSin(x)).Run()
}

// Slice returns a new operator node as a result of the gradfn.Slice function.
func Slice(x DualValue, fromRow, fromCol, toRow, toCol int) DualValue {
	return NewOperator(gradfn.NewSlice(x, fromRow, fromCol, toRow, toCol)).Run()
}

// Softmax returns a new operator node as a result of the gradfn.Softmax function.
func Softmax(x DualValue) DualValue {
	return NewOperator(gradfn.NewSoftmax(x)).Run()
}

// SoftPlus returns a new operator node as a result of the gradfn.SoftPlus function.
func SoftPlus(x, beta, threshold DualValue) DualValue {
	return NewOperator(gradfn.NewSoftPlus(x, beta, threshold)).Run()
}

// SoftShrink returns a new operator node as a result of the gradfn.SoftShrink function.
func SoftShrink(x, lambda DualValue) DualValue {
	return NewOperator(gradfn.NewSoftShrink(x, lambda)).Run()
}

// Softsign returns a new operator node as a result of the `SoftSign` function.
func Softsign(x DualValue) DualValue {
	return NewOperator(gradfn.NewSoftsign(x)).Run()
}

// SparseMax returns a new operator node as a result of the gradfn.SparseMax function.
func SparseMax(x DualValue) DualValue {
	return NewOperator(gradfn.NewSparseMax(x)).Run()
}

// SparseMaxLoss returns a new operator node as a result of the gradfn.SparseMaxLoss function.
func SparseMaxLoss(x DualValue) DualValue {
	return NewOperator(gradfn.NewSparseMaxLoss(x)).Run()
}

// Sqrt returns a new operator node as a result of the `Sqrt` function.
func Sqrt(x DualValue) DualValue {
	return NewOperator(gradfn.NewSqrt(x)).Run()
}

// Square returns a new operator node as a result of the gradfn.Prod(x, x) function.
func Square(x DualValue) DualValue {
	return NewOperator(gradfn.NewSquare(x)).Run()
}

// Stack returns a new operator node as a result of the gradfn.Stack function.
func Stack(xs ...DualValue) DualValue {
	return NewOperator(gradfn.NewStack(xs)).Run()
}

// Sub returns a new operator node as a result of the gradfn.Sub function.
func Sub(x1, x2 DualValue) DualValue {
	return NewOperator(gradfn.NewSub(x1, x2)).Run()
}

// SubScalar returns a new operator node as a result of the gradfn.SubScalar function.
func SubScalar(x1, x2 DualValue) DualValue {
	return NewOperator(gradfn.NewSubScalar(x1, x2)).Run()
}

// Swish returns a new operator node as a result of the gradfn.Swish function.
func Swish(x DualValue) DualValue {
	return NewOperator(gradfn.NewSwish(x)).Run()
}

// SwishB returns a new operator node as a result of the gradfn.SwishB function.
func SwishB(x, beta DualValue) DualValue {
	return NewOperator(gradfn.NewSwishB(x, beta)).Run()
}

// T returns a new operator node as a result of the fn.T function.
func T(x DualValue) DualValue {
	return NewOperator(gradfn.NewTranspose(x)).Run()
}

// Tan returns a new operator node as a result of the `Tan` function.
func Tan(x DualValue) DualValue {
	return NewOperator(gradfn.NewTan(x)).Run()
}

// Tanh returns a new operator node as a result of the `Tanh` function.
func Tanh(x DualValue) DualValue {
	return NewOperator(gradfn.NewTanh(x)).Run()
}

// Threshold returns a new operator node as a result of the gradfn.Threshold function.
func Threshold(x, threshold, k DualValue) DualValue {
	return NewOperator(gradfn.NewThreshold(x, threshold, k)).Run()
}

// Map returns a transformed version of xs with all its components modified according to the mapping function.
// It is useful for applying an operator to a sequence of nodes. Keep in mind that using this function has an overhead
// because of the callback, however insignificant compared to mathematical computations.
func Map(mapping func(DualValue) DualValue, xs []DualValue) []DualValue {
	ys := make([]DualValue, len(xs))
	for i, x := range xs {
		ys[i] = mapping(x)
	}
	return ys
}

// Map2 takes two arguments and applies a mapping function (that must take two arguments) to the items from the two node-slices in parallel.
// It panics if one slice is shorter than the other.
func Map2(mapping func(a DualValue, b DualValue) DualValue, xs1 []DualValue, xs2 []DualValue) []DualValue {
	if len(xs1) != len(xs2) {
		panic(fmt.Sprintf("ag: arguments must have the same size (%d != %d)", len(xs1), len(xs2)))
	}
	ys := make([]DualValue, len(xs1))
	for i, x1 := range xs1 {
		ys[i] = mapping(x1, xs2[i])
	}
	return ys
}

// Pad down/up samples the input to the given size.
func Pad(xs []DualValue, seqLen int, padding func(i int) DualValue) []DualValue {
	if len(xs) == seqLen {
		return xs
	}
	if len(xs) > seqLen {
		return xs[:seqLen]
	}
	padded := make([]DualValue, seqLen)
	copy(padded[:len(xs)], xs)
	for i := len(xs); i < len(padded); i++ {
		padded[i] = padding(i)
	}
	return padded
}

// SeparateMatrix returns a matrix of Node(s) represented as a slice of slice containing the elements extracted from the input.
// The dimensions of the resulting matrix are the same of the input.
func SeparateMatrix(x DualValue) [][]DualValue {
	shape := x.Value().Shape()
	rows, cols := shape[0], shape[1]
	ys := make([][]DualValue, rows)
	for i := range ys {
		row := make([]DualValue, cols)
		for j := range row {
			row[j] = At(x, i, j)
		}
		ys[i] = row
	}
	return ys
}

// SeparateVec returns a slice of Node(s) containing the elements extracted from the input.
// The size of the vector equals the number of input elements.
// You can think of this method as the inverse of the ag.Concat operator.
func SeparateVec(x DualValue) []DualValue {
	size := x.Value().Size()
	ys := make([]DualValue, size)
	for i := 0; i < size; i++ {
		ys[i] = At(x, i)
	}
	return ys
}

// SplitVec splits the x Node into multiple chunks.
func SplitVec(x DualValue, chunks int) []DualValue {
	if x.Value().Size()%chunks != 0 {
		panic("nn: incompatible chunks size")
	}
	l := 0
	size := int(math.Ceil(float64(x.Value().Size()) / float64(chunks)))
	ys := make([]DualValue, chunks)
	for i := 0; i < chunks; i++ {
		ys[i] = Slice(x, l, 0, l+size, 1)
		l += size
	}
	return ys
}

// Sum returns the value that describes the sum of the sample.
// It panics if the input is empty.
func Sum(xs ...DualValue) DualValue {
	sumVector := xs[0]
	for i := 1; i < len(xs); i++ {
		sumVector = Add(sumVector, xs[i])
	}
	return sumVector
}

// Mean returns the value that describes the average of the sample.
func Mean(xs []DualValue) DualValue {
	sumVector := xs[0]
	for i := 1; i < len(xs); i++ {
		sumVector = Add(sumVector, xs[i])
	}
	ln := sumVector.Value().NewScalar(float64(len(xs)))
	return DivScalar(sumVector, ln)
}

// Maximum returns the value that describes the maximum of the sample.
func Maximum(xs []DualValue) DualValue {
	maxVector := xs[0]
	for i := 1; i < len(xs); i++ {
		maxVector = Max(maxVector, xs[i])
	}
	return maxVector
}

// Minimum returns the value that describes the minimum of the sample.
func Minimum(xs []DualValue) DualValue {
	minVector := xs[0]
	for i := 1; i < len(xs); i++ {
		minVector = Min(minVector, xs[i])
	}
	return minVector
}

// BiLinear performs a bilinear transformation of the type (x_1 W x_2)
func BiLinear(w, x1, x2 DualValue) DualValue {
	return Mul(Mul(T(x1), w), x2)
}

// BiAffine performs a biaffine transformation.
func BiAffine(w, u, v, b, x1, x2 DualValue) DualValue {
	return Add(Add(Add(BiLinear(w, x1, x2), Mul(T(u), x1)), Mul(T(v), x2)), b)
}

// PositiveELU returns a new operator node as a result of ELU(x) + 1.
func PositiveELU(x DualValue) DualValue {
	one := x.Value().NewScalar(1)
	return AddScalar(ELU(x, one), one)
}

// LogSoftmax returns a new operator node as a result of Log(Softmax(x)).
func LogSoftmax(x DualValue) DualValue {
	return Log(Softmax(x))
}

// LogSumExp "trick" computes the log of the sum of exponentials of input elements.
// When the input is one, this must be a vector. Alternatively, the calculation
// is conducted on a list of scalars.
func LogSumExp(xs ...DualValue) DualValue {
	if len(xs) == 1 {
		x := xs[0]
		max := ReduceMax(x)
		sum := ReduceSum(Exp(SubScalar(x, max)))
		return Add(max, Log(sum))
	}

	max := ScalarMax(xs)
	var sum DualValue
	for _, v := range xs {
		sum = Add(sum, Exp(Sub(v, max)))
	}
	return Add(max, Log(sum))
}

// RowViews calls RowView for each row of x, returning a new slice
// of row-view Nodes.
func RowViews(x DualValue) []DualValue {
	ys := make([]DualValue, x.Value().Shape()[0])
	for i := range ys {
		ys[i] = RowView(x, i)
	}
	return ys
}

// ColViews calls ColView for each column of x, returning a new slice
// of column-view Nodes.
func ColViews(x DualValue) []DualValue {
	ys := make([]DualValue, x.Value().Shape()[1])
	for i := range ys {
		ys[i] = ColView(x, i)
	}
	return ys
}
