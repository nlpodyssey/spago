// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"fmt"
	"math"
	"sync"

	"github.com/nlpodyssey/spago/fn"
)

// Abs returns a new operator node as a result of the `Abs` function.
func Abs(x DualValue) DualValue {
	return NewOperator(fn.NewAbs(x))
}

// Add returns a new operator node as a result of the fn.Add function.
// As special case, the first node may be null.
// This help to keep the code as concise as possible e.g. during accumulation.
func Add(x1 DualValue, x2 DualValue) DualValue {
	if x1 == nil {
		return Identity(x2) // return a copy of `x2` as is
	}
	return NewOperator(fn.NewAdd(x1, x2))
}

// AddScalar returns a new operator node as a result of the fn.AddScalar function.
func AddScalar(x1, x2 DualValue) DualValue {
	return NewOperator(fn.NewAddScalar(x1, x2))
}

// Affine returns a new operator node as a result of the fn.Affine function.
func Affine(b, w1, x1 DualValue, wxPairs ...DualValue) DualValue {
	return NewOperator(fn.NewAffine(b, w1, x1, wxPairs...))
}

// AppendRows returns a new operator node as a result of the fn.AppendRows function.
func AppendRows(x DualValue, vs ...DualValue) DualValue {
	return NewOperator(fn.NewAppendRows(x, vs...))
}

// At returns a new operator node as a result of the fn.At function.
func At(x DualValue, i int, j int) DualValue {
	return NewOperator(fn.NewAt(x, i, j))
}

// AtVec returns a new operator node as a result of the fn.AtVec function.
func AtVec(x DualValue, i int) DualValue {
	return NewOperator(fn.NewAtVec(x, i))
}

// CELU returns a new operator node as a result of the fn.CELU function.
func CELU(x, alpha DualValue) DualValue {
	return NewOperator(fn.NewCELU(x, alpha))
}

// ColView returns a new operator node as a result of the fn.ColView function.
func ColView(x DualValue, column int) DualValue {
	return NewOperator(fn.NewColView(x, column))
}

// Concat returns a new operator node as a result of the fn.Concat function.
func Concat(xs ...DualValue) DualValue {
	return NewOperator(fn.NewConcat(xs))
}

// Cos returns a new operator node as a result of the `Cos` function.
func Cos(x DualValue) DualValue {
	return NewOperator(fn.NewCos(x))
}

// Div returns a new operator node as a result of the fn.Div function.
func Div(x1, x2 DualValue) DualValue {
	return NewOperator(fn.NewDiv(x1, x2))
}

// DivScalar returns a new operator node as a result of the fn.DivScalar function.
func DivScalar(x1, x2 DualValue) DualValue {
	return NewOperator(fn.NewDivScalar(x1, x2))
}

// Dot returns a new operator node as a result of the fn.Dot function.
func Dot(x1, x2 DualValue) DualValue {
	return NewOperator(fn.NewDot(x1, x2))
}

// DropoutFunc returns a function to create a Dropout operator working with the given dropout probability.
func DropoutFunc(p float64) func(x DualValue) DualValue {
	return func(x DualValue) DualValue {
		if p == 0.0 {
			return x
		}
		return NewOperator(fn.NewDropout(x, p, globalGenerator))
	}
}

// Dropout returns a new operator node as a result of the fn.Dropout function.
// If the dropout probability is zero, the operator will not be created,
// so the input itself is returned directly.
func Dropout(x DualValue, p float64) DualValue {
	if p == 0.0 {
		return x
	}
	return NewOperator(fn.NewDropout(x, p, globalGenerator))
}

// ELU returns a new operator node as a result of the fn.ELU function.
func ELU(x, alpha DualValue) DualValue {
	return NewOperator(fn.NewELU(x, alpha))
}

// Exp returns a new operator node as a result of the `Exp` function.
func Exp(x DualValue) DualValue {
	return NewOperator(fn.NewExp(x))
}

// Flatten returns a new operator node as a result of the fn.Flatten function.
func Flatten(x DualValue) DualValue {
	return NewOperator(fn.NewFlatten(x))
}

// GELU returns a new operator node as a result of the fn.GELU function.
func GELU(x DualValue) DualValue {
	return NewOperator(fn.NewGELU(x))
}

// HardSigmoid returns a new operator node as a result of the `HardSigmoid` function.
func HardSigmoid(x DualValue) DualValue {
	return NewOperator(fn.NewHardSigmoid(x))
}

// HardTanh returns a new operator node as a result of the `HardTanh` function.
func HardTanh(x DualValue) DualValue {
	return NewOperator(fn.NewHardTanh(x))
}

// Identity returns a new operator node as a result of the fn.Identity function.
func Identity(x DualValue) DualValue {
	return NewOperator(fn.NewIdentity(x))
}

// LeakyReLU returns a new operator node as a result of the fn.LeakyReLU function.
func LeakyReLU(x, alpha DualValue) DualValue {
	return NewOperator(fn.NewLeakyReLU(x, alpha))
}

// Log returns a new operator node as a result of the `Log` function.
func Log(x DualValue) DualValue {
	return NewOperator(fn.NewLog(x))
}

// Max returns a new operator node as a result of the fn.Max function.
func Max(x1, x2 DualValue) DualValue {
	return NewOperator(fn.NewMax(x1, x2))
}

// MaxPooling returns a new operator node as a result of the fn.MaxPooling function.
func MaxPooling(x DualValue, rows, columns int) DualValue {
	return NewOperator(fn.NewMaxPooling(x, rows, columns))
}

// Min returns a new operator node as a result of the fn.Min function.
func Min(x1, x2 DualValue) DualValue {
	return NewOperator(fn.NewMin(x1, x2))
}

// Mish returns a new operator node as a result of the `Mish` function.
func Mish(x DualValue) DualValue {
	return NewOperator(fn.NewMish(x))
}

// Mul returns a new operator node as a result of the fn.Mul function.
func Mul(x1, x2 DualValue) DualValue {
	return NewOperator(fn.NewMul(x1, x2))
}

func MulT(x1, x2 DualValue) DualValue {
	return NewOperator(fn.NewMulT(x1, x2))
}

// Neg returns a new operator node as a result of the `Neg` function.
func Neg(x DualValue) DualValue {
	return NewOperator(fn.NewNeg(x))
}

// Pow returns a new operator node as a result of the fn.Pow function.
func Pow(x DualValue, power float64) DualValue {
	return NewOperator(fn.NewPow(x, power))
}

// Prod returns a new operator node as a result of the fn.Prod function.
func Prod(x1, x2 DualValue) DualValue {
	return NewOperator(fn.NewProd(x1, x2))
}

// ProdScalar returns a new operator node as a result of the fn.ProdScalar function.
func ProdScalar(x1, x2 DualValue) DualValue {
	return NewOperator(fn.NewProdScalar(x1, x2))
}

// Reciprocal returns a new operator node as a result of the `Reciprocal` function.
func Reciprocal(x DualValue) DualValue {
	return NewOperator(fn.NewReciprocal(x))
}

// ReduceMax returns a new operator node as a result of the fn.ReduceMax function.
func ReduceMax(x DualValue) DualValue {
	return NewOperator(fn.NewReduceMax(x))
}

// ReduceMean returns a new operator node as a result of the fn.ReduceMean function.
func ReduceMean(x DualValue) DualValue {
	return NewOperator(fn.NewReduceMean(x))
}

// ReduceSum returns a new operator node as a result of the fn.ReduceSum function.
func ReduceSum(x DualValue) DualValue {
	return NewOperator(fn.NewReduceSum(x))
}

// ReLU returns a new operator node as a result of the `ReLU` function.
func ReLU(x DualValue) DualValue {
	return NewOperator(fn.NewReLU(x))
}

// Reshape returns a new operator node as a result of the fn.Reshape function.
func Reshape(x DualValue, rows, columns int) DualValue {
	return NewOperator(fn.NewReshape(x, rows, columns))
}

// ReverseSub returns a new operator node as a result of the fn.ReverseSub function.
func ReverseSub(x1, x2 DualValue) DualValue {
	return NewOperator(fn.NewReverseSubScalar(x1, x2))
}

// RotateR performs the right circular shift.
// `i` is the number of places by which the elements are shifted.
func RotateR(x DualValue, i int) DualValue {
	return NewOperator(fn.NewRotateR(x, i))
}

// RowView returns a new operator node as a result of the fn.RowView function.
func RowView(x DualValue, row int) DualValue {
	return NewOperator(fn.NewRowView(x, row))
}

// ScalarMax returns a new operator node as a result of the fn.ScalarMax function.
func ScalarMax(xs []DualValue) DualValue {
	return NewOperator(fn.NewScalarMax(xs))
}

// SELU returns a new operator node as a result of the fn.SELU function.
func SELU(x, alpha DualValue, scale DualValue) DualValue {
	return NewOperator(fn.NewSELU(x, alpha, scale))
}

// Sigmoid returns a new operator node as a result of the `Sigmoid` function.
func Sigmoid(x DualValue) DualValue {
	return NewOperator(fn.NewSigmoid(x))
}

// SiLU returns a new operator node as a result of the fn.SiLU function.
func SiLU(x DualValue) DualValue {
	return NewOperator(fn.NewSiLU(x))
}

// Sin returns a new operator node as a result of the `Sin` function.
func Sin(x DualValue) DualValue {
	return NewOperator(fn.NewSin(x))
}

// Slice returns a new operator node as a result of the fn.Slice function.
func Slice(x DualValue, fromRow, fromCol, toRow, toCol int) DualValue {
	return NewOperator(fn.NewSlice(x, fromRow, fromCol, toRow, toCol))
}

// Softmax returns a new operator node as a result of the fn.Softmax function.
func Softmax(x DualValue) DualValue {
	return NewOperator(fn.NewSoftmax(x))
}

// SoftPlus returns a new operator node as a result of the fn.SoftPlus function.
func SoftPlus(x, beta, threshold DualValue) DualValue {
	return NewOperator(fn.NewSoftPlus(x, beta, threshold))
}

// SoftShrink returns a new operator node as a result of the fn.SoftShrink function.
func SoftShrink(x, lambda DualValue) DualValue {
	return NewOperator(fn.NewSoftShrink(x, lambda))
}

// Softsign returns a new operator node as a result of the `SoftSign` function.
func Softsign(x DualValue) DualValue {
	return NewOperator(fn.NewSoftsign(x))
}

// SparseMax returns a new operator node as a result of the fn.SparseMax function.
func SparseMax(x DualValue) DualValue {
	return NewOperator(fn.NewSparseMax(x))
}

// SparseMaxLoss returns a new operator node as a result of the fn.SparseMaxLoss function.
func SparseMaxLoss(x DualValue) DualValue {
	return NewOperator(fn.NewSparseMaxLoss(x))
}

// Sqrt returns a new operator node as a result of the `Sqrt` function.
func Sqrt(x DualValue) DualValue {
	return NewOperator(fn.NewSqrt(x))
}

// Square returns a new operator node as a result of the fn.Prod(x, x) function.
func Square(x DualValue) DualValue {
	return NewOperator(fn.NewSquare(x))
}

// Stack returns a new operator node as a result of the fn.Stack function.
func Stack(xs ...DualValue) DualValue {
	return NewOperator(fn.NewStack(xs))
}

// Sub returns a new operator node as a result of the fn.Sub function.
func Sub(x1, x2 DualValue) DualValue {
	return NewOperator(fn.NewSub(x1, x2))
}

// SubScalar returns a new operator node as a result of the fn.SubScalar function.
func SubScalar(x1, x2 DualValue) DualValue {
	return NewOperator(fn.NewSubScalar(x1, x2))
}

// Swish returns a new operator node as a result of the fn.Swish function.
func Swish(x DualValue) DualValue {
	return NewOperator(fn.NewSwish(x))
}

// SwishB returns a new operator node as a result of the fn.SwishB function.
func SwishB(x, beta DualValue) DualValue {
	return NewOperator(fn.NewSwishB(x, beta))
}

// T returns a new operator node as a result of the fn.T function.
func T(x DualValue) DualValue {
	return NewOperator(fn.NewTranspose(x))
}

// Tan returns a new operator node as a result of the `Tan` function.
func Tan(x DualValue) DualValue {
	return NewOperator(fn.NewTan(x))
}

// Tanh returns a new operator node as a result of the `Tanh` function.
func Tanh(x DualValue) DualValue {
	return NewOperator(fn.NewTanh(x))
}

// Threshold returns a new operator node as a result of the fn.Threshold function.
func Threshold(x, threshold, k DualValue) DualValue {
	return NewOperator(fn.NewThreshold(x, threshold, k))
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

// MapConcurrent is the concurrent version of Map.
func MapConcurrent(mapping func(DualValue) DualValue, xs []DualValue) []DualValue {
	var wg sync.WaitGroup
	wg.Add(len(xs))
	ys := make([]DualValue, len(xs))
	for i, x := range xs {
		i, x := i, x
		go func() {
			ys[i] = mapping(x)
			wg.Done()
		}()
	}
	wg.Wait()
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

// Map2Concurrent is the concurrent version of Map2.
func Map2Concurrent(mapping func(a DualValue, b DualValue) DualValue, xs1 []DualValue, xs2 []DualValue) []DualValue {
	if len(xs1) != len(xs2) {
		panic(fmt.Sprintf("ag: arguments must have the same size (%d != %d)", len(xs1), len(xs2)))
	}
	var wg sync.WaitGroup
	wg.Add(len(xs1))
	ys := make([]DualValue, len(xs1))
	for i, x1 := range xs1 {
		i, x1 := i, x1
		go func() {
			ys[i] = mapping(x1, xs2[i])
			wg.Done()
		}()
	}
	wg.Wait()
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
	rows, cols := x.Value().Dims()
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
		ys[i] = AtVec(x, i)
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
	var sum Node
	for _, v := range xs {
		sum = Add(sum, Exp(Sub(v, max)))
	}
	return Add(max, Log(sum))
}

// RowViews calls RowView for each row of x, returning a new slice
// of row-view Nodes.
func RowViews(x DualValue) []DualValue {
	ys := make([]DualValue, x.Value().Rows())
	for i := range ys {
		ys[i] = RowView(x, i)
	}
	return ys
}

// ColViews calls ColView for each column of x, returning a new slice
// of column-view Nodes.
func ColViews(x DualValue) []DualValue {
	ys := make([]DualValue, x.Value().Columns())
	for i := range ys {
		ys[i] = ColView(x, i)
	}
	return ys
}
