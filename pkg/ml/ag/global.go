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

// GlobalGraphDType is the data type used for the global Graph.
type GlobalGraphDType = float32

var globalGraph = NewGraph(Rand(rand.NewLockedRand[GlobalGraphDType](42)))

// GetGlobalGraph returns the global graph.
// Although technically you could reassign the returned graph, please do not do so; imagine that its reference is immutable.
// Otherwise you are likely to generate inconsistent computations.
// To clean the global graph, you can use ClearGlobalGraph() or ClearGlobalGraphForReuse().
func GetGlobalGraph() *Graph[GlobalGraphDType] {
	return globalGraph
}

// ClearGlobalGraph clears the global graph. This is a destructive operation.
// See Graph.Clear() for more information.
func ClearGlobalGraph() {
	globalGraph.Clear()
}

// ClearGlobalGraphForReuse does the same thing as ClearGlobalGraph(), with the difference that the
// graph structure is maintained.
// See Graph.ClearForReuse() for more information.
func ClearGlobalGraphForReuse() {
	globalGraph.ClearForReuse()
}

// ZeroGrad sets the gradients of all nodes of the global graph to zero.
func ZeroGrad() {
	globalGraph.ZeroGrad()
}

// NewVariable creates and returns a new node.
func NewVariable(value mat.Matrix[GlobalGraphDType], requiresGrad bool) Node[GlobalGraphDType] {
	return globalGraph.NewVariable(value, requiresGrad)
}

// NewScalar creates a variable node that doesn't require gradients.
func NewScalar(value GlobalGraphDType) Node[GlobalGraphDType] {
	return globalGraph.NewScalar(value)
}

// NewOperator creates a new operator along with its forward pass.
func NewOperator(f fn.Function[GlobalGraphDType], operands ...Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.NewOperator(f, operands...)
}

// NewWrap creates a new wrapper Node for the given value, attaching it to
// the global graph.
func NewWrap(value GradValue[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.NewWrap(value)
}

// NewWrapNoGrad is similar to NewWrap, but it disables automatic
// differentiation on the new node.
func NewWrapNoGrad(value GradValue[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.NewWrapNoGrad(value)
}

// ReplaceValue replaces the current value of a variable Node with the given value,
// on the global graph. It panics if node is not a variable.
func ReplaceValue(node Node[GlobalGraphDType], value mat.Matrix[GlobalGraphDType]) {
	globalGraph.ReplaceValue(node, value)
}

// IncTimeStep increments the value of the global graph's TimeStep by one.
func IncTimeStep() {
	globalGraph.IncTimeStep()
}

// TimeStep is an integer value associated with the global graph, which can be useful
// to perform truncated back propagation.
func TimeStep() int {
	return globalGraph.TimeStep()
}

// Nodes returns the nodes of the graph.
func Nodes() []Node[GlobalGraphDType] {
	return globalGraph.Nodes()
}

// Forward computes the results of the entire global raph.
func Forward(opts ...ForwardOption[GlobalGraphDType]) {
	globalGraph.Forward(opts...)
}

// Backward performs the back-propagation.
// See Graph.Backward() for more information.
func Backward(node Node[GlobalGraphDType], opts ...BackwardOption[GlobalGraphDType]) {
	globalGraph.Backward(node, opts...)
}

// BackwardAll performs full back-propagation from the last node of the graph.
// It requires the root nodes to have assigned gradients already.
func BackwardAll() {
	globalGraph.BackwardAll()
}

// Invoke returns a new node as a result of the application of the input operator.
func Invoke(operator OpName, xs ...Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Invoke(operator, xs...)
}

// Identity returns a new operator node as a result of the fn.Identity function.
func Identity(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Identity(x)
}

// Dropout returns a new operator node as a result of the fn.Dropout function.
func Dropout(x Node[GlobalGraphDType], p GlobalGraphDType) Node[GlobalGraphDType] {
	return globalGraph.Dropout(x, p)
}

// AtVec returns a new operator node as a result of the fn.AtVec function.
func AtVec(x Node[GlobalGraphDType], i int) Node[GlobalGraphDType] {
	return globalGraph.AtVec(x, i)
}

// At returns a new operator node as a result of the fn.At function.
func At(x Node[GlobalGraphDType], i int, j int) Node[GlobalGraphDType] {
	return globalGraph.At(x, i, j)
}

// Add returns a new operator node as a result of the fn.Add function.
// The first node may be null. This help to keep the code as concise as possible e.g. during accumulation.
func Add(x1, x2 Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Add(x1, x2)
}

// Sub returns a new operator node as a result of the fn.Sub function.
func Sub(x1, x2 Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Sub(x1, x2)
}

// SubScalar returns a new operator node as a result of the fn.SubScalar function.
func SubScalar(x1, x2 Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.SubScalar(x1, x2)
}

// AddScalar returns a new operator node as a result of the fn.AddScalar function.
func AddScalar(x1, x2 Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.AddScalar(x1, x2)
}

// ReverseSub returns a new operator node as a result of the fn.ReverseSub function.
func ReverseSub(x1, x2 Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.ReverseSub(x1, x2)
}

// Prod returns a new operator node as a result of the fn.Prod function.
func Prod(x1, x2 Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Prod(x1, x2)
}

// Div returns a new operator node as a result of the fn.Div function.
func Div(x1, x2 Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Div(x1, x2)
}

// ProdScalar returns a new operator node as a result of the fn.ProdScalar function.
func ProdScalar(x1, x2 Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.ProdScalar(x1, x2)
}

// DivScalar returns a new operator node as a result of the fn.DivScalar function.
func DivScalar(x1, x2 Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.DivScalar(x1, x2)
}

// Mul returns a new operator node as a result of the fn.Mul function.
func Mul(x1, x2 Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Mul(x1, x2)
}

// Dot returns a new operator node as a result of the fn.Dot function.
func Dot(x1, x2 Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Dot(x1, x2)
}

// Max returns a new operator node as a result of the fn.Max function.
func Max(x1, x2 Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Max(x1, x2)
}

// Min returns a new operator node as a result of the fn.Min function.
func Min(x1, x2 Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Min(x1, x2)
}

// Reshape returns a new operator node as a result of the fn.Reshape function.
func Reshape(x Node[GlobalGraphDType], rows, columns int) Node[GlobalGraphDType] {
	return globalGraph.Reshape(x, rows, columns)
}

// MaxPooling returns a new operator node as a result of the fn.MaxPooling function.
func MaxPooling(x Node[GlobalGraphDType], rows, columns int) Node[GlobalGraphDType] {
	return globalGraph.MaxPooling(x, rows, columns)
}

// View returns a new operator node as a result of the fn.View function.
func View(x Node[GlobalGraphDType], row, column, xStride, yStride int) Node[GlobalGraphDType] {
	return globalGraph.View(x, row, column, xStride, yStride)
}

// RowView returns a new operator node as a result of the fn.RowView function.
func RowView(x Node[GlobalGraphDType], row int) Node[GlobalGraphDType] {
	return globalGraph.RowView(x, row)
}

// ColView returns a new operator node as a result of the fn.ColView function.
func ColView(x Node[GlobalGraphDType], column int) Node[GlobalGraphDType] {
	return globalGraph.ColView(x, column)
}

// RotateR performs the right circular shift.
// `i` is the number of places by which the elements are shifted.
func RotateR(x Node[GlobalGraphDType], i int) Node[GlobalGraphDType] {
	return globalGraph.RotateR(x, i)
}

// Vec returns a new operator node as a result of the fn.Vec function.
func Vec(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Vec(x)
}

// T returns a new operator node as a result of the fn.T function.
func T(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.T(x)
}

// Square returns a new operator node as a result of the fn.Prod(x, x) function.
func Square(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Square(x)
}

// Pow returns a new operator node as a result of the fn.Pow function.
func Pow(x Node[GlobalGraphDType], power GlobalGraphDType) Node[GlobalGraphDType] {
	return globalGraph.Pow(x, power)
}

// Sqrt returns a new operator node as a result of the `Sqrt` function.
func Sqrt(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Sqrt(x)
}

// Tan returns a new operator node as a result of the `Tan` function.
func Tan(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Tan(x)
}

// Tanh returns a new operator node as a result of the `Tanh` function.
func Tanh(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Tanh(x)
}

// Sigmoid returns a new operator node as a result of the `Sigmoid` function.
func Sigmoid(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Sigmoid(x)
}

// HardSigmoid returns a new operator node as a result of the `HardSigmoid` function.
func HardSigmoid(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.HardSigmoid(x)
}

// HardTanh returns a new operator node as a result of the `HardTanh` function.
func HardTanh(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.HardTanh(x)
}

// Softsign returns a new operator node as a result of the `SoftSign` function.
func Softsign(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Softsign(x)
}

// ReLU returns a new operator node as a result of the `ReLU` function.
func ReLU(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.ReLU(x)
}

// CELU returns a new operator node as a result of the fn.CELU function.
func CELU(x Node[GlobalGraphDType], alpha Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.CELU(x, alpha)
}

// GELU returns a new operator node as a result of the fn.GELU function.
func GELU(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.GELU(x)
}

// ELU returns a new operator node as a result of the fn.ELU function.
func ELU(x Node[GlobalGraphDType], alpha Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.ELU(x, alpha)
}

// PositiveELU returns a new operator node as a result of ELU(x, 1.0) + 1.
func PositiveELU(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.PositiveELU(x)
}

// SwishB returns a new operator node as a result of the fn.SwishB function.
func SwishB(x Node[GlobalGraphDType], beta Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.SwishB(x, beta)
}

// Swish returns a new operator node as a result of the fn.Swish function.
func Swish(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Swish(x)
}

// SiLU returns a new operator node as a result of the fn.SiLU function.
func SiLU(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.SiLU(x)
}

// Mish returns a new operator node as a result of the `Mish` function.
func Mish(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Mish(x)
}

// LeakyReLU returns a new operator node as a result of the fn.LeakyReLU function.
func LeakyReLU(x Node[GlobalGraphDType], alpha Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.LeakyReLU(x, alpha)
}

// SELU returns a new operator node as a result of the fn.SELU function.
func SELU(x, alpha, scale Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.SELU(x, alpha, scale)
}

// SoftPlus returns a new operator node as a result of the fn.SoftPlus function.
func SoftPlus(x, beta, threshold Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.SoftPlus(x, beta, threshold)
}

// SoftShrink returns a new operator node as a result of the fn.SoftShrink function.
func SoftShrink(x, lambda Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.SoftShrink(x, lambda)
}

// Threshold returns a new operator node as a result of the fn.Threshold function.
func Threshold(x, threshold, k Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Threshold(x, threshold, k)
}

// Softmax returns a new operator node as a result of the fn.Softmax function.
func Softmax(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Softmax(x)
}

// LogSoftmax returns a new operator node as a result of Log(Softmax(x)).
func LogSoftmax(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.LogSoftmax(x)
}

// SparseMax returns a new operator node as a result of the fn.SparseMax function.
func SparseMax(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.SparseMax(x)
}

// SparseMaxLoss returns a new operator node as a result of the fn.SparseMaxLoss function.
func SparseMaxLoss(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.SparseMaxLoss(x)
}

// Sin returns a new operator node as a result of the `Sin` function.
func Sin(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Sin(x)
}

// Cos returns a new operator node as a result of the `Cos` function.
func Cos(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Cos(x)
}

// Exp returns a new operator node as a result of the `Exp` function.
func Exp(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Exp(x)
}

// Log returns a new operator node as a result of the `Log` function.
func Log(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Log(x)
}

// Abs returns a new operator node as a result of the `Abs` function.
func Abs(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Abs(x)
}

// Neg returns a new operator node as a result of the `Neg` function.
func Neg(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Neg(x)
}

// Reciprocal returns a new operator node as a result of the `Reciprocal` function.
func Reciprocal(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Reciprocal(x)
}

// ReduceSum returns a new operator node as a result of the fn.ReduceSum function.
func ReduceSum(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.ReduceSum(x)
}

// ReduceMean returns a new operator node as a result of the fn.ReduceMean function.
func ReduceMean(x Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.ReduceMean(x)
}

// Sum returns the value that describes the sum of the sample.
func Sum(xs ...Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Sum(xs...)
}

// Mean returns the value that describes the average of the sample.
func Mean(xs []Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Mean(xs)
}

// Concat returns a new operator node as a result of the fn.Concat function.
func Concat(xs ...Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Concat(xs...)
}

// Stack returns a new operator node as a result of the fn.Stack function.
func Stack(xs ...Node[GlobalGraphDType]) Node[GlobalGraphDType] {
	return globalGraph.Stack(xs...)
}
