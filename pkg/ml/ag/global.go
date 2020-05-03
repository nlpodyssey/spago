// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag/fn"
)

/*
 * Top-level convenience functions
 */

var globalGraph = NewGraph(Rand(rand.NewLockedRand(42)))

// GetGlobalGraph returns the global graph.
// Although technically you could reassign the returned graph, please do not do so; imagine that its reference is immutable.
// Otherwise you are likely to generate inconsistent computations.
// To clean the global graph, you can use ClearGlobalGraph() or ClearGlobalGraphForReuse().
func GetGlobalGraph() *Graph {
	return globalGraph
}

func ClearGlobalGraph() {
	globalGraph.Clear()
}

func ClearGlobalGraphForReuse() {
	globalGraph.ClearForReuse()
}

func ZeroGrad() {
	globalGraph.ZeroGrad()
}

func NewVariable(value mat.Matrix, requiresGrad bool) Node {
	return globalGraph.NewVariable(value, requiresGrad)
}

func NewScalar(value float64) Node {
	return globalGraph.NewScalar(value)
}

func NewOperator(f fn.Function, operands ...Node) Node {
	return globalGraph.NewOperator(f, operands...)
}

func NewWrap(value GradValue) Node {
	return globalGraph.NewWrap(value)
}

func NewWrapNoGrad(value GradValue) Node {
	return globalGraph.NewWrapNoGrad(value)
}

func ReplaceValue(node Node, value mat.Matrix) {
	globalGraph.ReplaceValue(node, value)
}

func IncTimeStep() {
	globalGraph.IncTimeStep()
}

func TimeStep() int {
	return globalGraph.TimeStep()
}

func Forward(opts ...ForwardOption) {
	globalGraph.Forward(opts...)
}

func Backward(node Node, opts ...BackwardOption) {
	globalGraph.Backward(node, opts...)
}

func BackwardAll() {
	globalGraph.BackwardAll()
}

// Invoke
func Invoke(operator OpName, xs ...Node) Node {
	return globalGraph.Invoke(operator, xs...)
}

// Identity
func Identity(x Node) Node {
	return globalGraph.Identity(x)
}

// Dropout
func Dropout(x Node, p float64) Node {
	return globalGraph.Dropout(x, p)
}

// AtVec
func AtVec(x Node, i int) Node {
	return globalGraph.AtVec(x, i)
}

// At
func At(x Node, i int, j int) Node {
	return globalGraph.At(x, i, j)
}

// Add
func Add(x1 Node, x2 Node) Node {
	return globalGraph.Add(x1, x2)
}

// Sub
func Sub(x1 Node, x2 Node) Node {
	return globalGraph.Sub(x1, x2)
}

// SubScalar
func SubScalar(x1 Node, x2 Node) Node {
	return globalGraph.SubScalar(x1, x2)
}

// AddScalar
func AddScalar(x1 Node, x2 Node) Node {
	return globalGraph.AddScalar(x1, x2)
}

// ReverseSub
func ReverseSub(x1 Node, x2 Node) Node {
	return globalGraph.ReverseSub(x1, x2)
}

// Prod
func Prod(x1 Node, x2 Node) Node {
	return globalGraph.Prod(x1, x2)
}

// Div
func Div(x1 Node, x2 Node) Node {
	return globalGraph.Div(x1, x2)
}

// ProdScalar
func ProdScalar(x1 Node, x2 Node) Node {
	return globalGraph.ProdScalar(x1, x2)
}

// DivScalar
func DivScalar(x1 Node, x2 Node) Node {
	return globalGraph.DivScalar(x1, x2)
}

// Mul
func Mul(x1 Node, x2 Node) Node {
	return globalGraph.Mul(x1, x2)
}

// Dot
func Dot(x1 Node, x2 Node) Node {
	return globalGraph.Dot(x1, x2)
}

// Max
func Max(x1 Node, x2 Node) Node {
	return globalGraph.Max(x1, x2)
}

// Min
func Min(x1 Node, x2 Node) Node {
	return globalGraph.Min(x1, x2)
}

// Reshape
func Reshape(x Node, rows, columns int) Node {
	return globalGraph.Reshape(x, rows, columns)
}

// MaxPooling
func MaxPooling(x Node, rows, columns int) Node {
	return globalGraph.MaxPooling(x, rows, columns)
}

// View
func View(x Node, row, column, xStride, yStride int) Node {
	return globalGraph.View(x, row, column, xStride, yStride)
}

// RowView
func RowView(x Node, row int) Node {
	return globalGraph.RowView(x, row)
}

// ColView
func ColView(x Node, column int) Node {
	return globalGraph.ColView(x, column)
}

// Vec
func Vec(x Node) Node {
	return globalGraph.Vec(x)
}

// T
func T(x Node) Node {
	return globalGraph.T(x)
}

// Square
func Square(x Node) Node {
	return globalGraph.Square(x)
}

// Pow
func Pow(x Node, power float64) Node {
	return globalGraph.Pow(x, power)
}

// Sqrt
func Sqrt(x Node) Node {
	return globalGraph.Sqrt(x)
}

// Tan
func Tan(x Node) Node {
	return globalGraph.Tan(x)
}

// Tanh
func Tanh(x Node) Node {
	return globalGraph.Tanh(x)
}

// Sigmoid
func Sigmoid(x Node) Node {
	return globalGraph.Sigmoid(x)
}

// HardSigmoid
func HardSigmoid(x Node) Node {
	return globalGraph.HardSigmoid(x)
}

// HardTanh
func HardTanh(x Node) Node {
	return globalGraph.HardTanh(x)
}

// Softsign
func Softsign(x Node) Node {
	return globalGraph.Softsign(x)
}

// ReLU
func ReLU(x Node) Node {
	return globalGraph.ReLU(x)
}

// CeLU
func CeLU(x Node, alpha Node) Node {
	return globalGraph.CeLU(x, alpha)
}

// ELU
func ELU(x Node, alpha Node) Node {
	return globalGraph.ELU(x, alpha)
}

// Swish
func Swish(x Node, beta Node) Node {
	return globalGraph.Swish(x, beta)
}

// Mish
func Mish(x Node) Node {
	return globalGraph.Mish(x)
}

// LeakyReLU
func LeakyReLU(x Node, alpha Node) Node {
	return globalGraph.LeakyReLU(x, alpha)
}

// SeLU
func SeLU(x Node, alpha Node, scale Node) Node {
	return globalGraph.SeLU(x, alpha, scale)
}

// SoftPlus
func SoftPlus(x Node, beta Node, threshold Node) Node {
	return globalGraph.SoftPlus(x, beta, threshold)
}

// SoftShrink
func SoftShrink(x Node, lambda Node) Node {
	return globalGraph.SoftShrink(x, lambda)
}

// Threshold
func Threshold(x Node, threshold Node, k Node) Node {
	return globalGraph.Threshold(x, threshold, k)
}

// Softmax
func Softmax(x Node) Node {
	return globalGraph.Softmax(x)
}

// Sin
func Sin(x Node) Node {
	return globalGraph.Sin(x)
}

// Cos
func Cos(x Node) Node {
	return globalGraph.Cos(x)
}

// Exp
func Exp(x Node) Node {
	return globalGraph.Exp(x)
}

// Log
func Log(x Node) Node {
	return globalGraph.Log(x)
}

// Abs
func Abs(x Node) Node {
	return globalGraph.Abs(x)
}

// Neg
func Neg(x Node) Node {
	return globalGraph.Neg(x)
}

// Reciprocal
func Reciprocal(x Node) Node {
	return globalGraph.Reciprocal(x)
}

// ReduceSum
func ReduceSum(x Node) Node {
	return globalGraph.ReduceSum(x)
}

// ReduceMean
func ReduceMean(x Node) Node {
	return globalGraph.ReduceMean(x)
}

// Concat
func Concat(xs ...Node) Node {
	return globalGraph.Concat(xs...)
}

// Stack
func Stack(xs ...Node) Node {
	return globalGraph.Stack(xs...)
}
