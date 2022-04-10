// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package losses

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

// MAE measures the mean absolute error (a.k.a. L1 Loss) between each element in the input x and target y.
func MAE[T mat.DType](x ag.Node[T], y ag.Node[T], reduceMean bool) ag.Node[T] {
	loss := ag.Abs(ag.Sub(x, y))
	if reduceMean {
		return ag.ReduceMean(loss)
	}
	return ag.ReduceSum(loss)
}

// MSE measures the mean squared error (squared L2 norm) between each element in the input x and target y.
func MSE[T mat.DType](x ag.Node[T], y ag.Node[T], reduceMean bool) ag.Node[T] {
	loss := ag.ProdScalar(ag.Square(ag.Sub(x, y)), ag.Constant[T](0.5))
	if reduceMean {
		return ag.ReduceMean(loss)
	}
	return ag.ReduceSum(loss)
}

// NLL returns the loss of the input x respect to the target y.
// The target is expected to be a one-hot vector.
func NLL[T mat.DType](x ag.Node[T], y ag.Node[T]) ag.Node[T] {
	return ag.Neg(ag.ReduceSum(ag.Prod(y, ag.Log(x))))
}

// CrossEntropy implements a cross-entropy loss function.
// x is the raw scores for each class (logits).
// c is the index of the gold class.
func CrossEntropy[T mat.DType](x ag.Node[T], c int) ag.Node[T] {
	return ag.Add(ag.Neg(ag.AtVec(x, c)), ag.LogSumExp(x))
}

// WeightedCrossEntropy implements a weighted cross-entropy loss function.
// x is the raw scores for each class (logits).
// c is the index of the gold class.
// This function is scaled by a weighting factor weights[class] ∈ [0,1]
func WeightedCrossEntropy[T mat.DType](weights []T) func(x ag.Node[T], c int) ag.Node[T] {
	return func(x ag.Node[T], c int) ag.Node[T] {
		return ag.ProdScalar(CrossEntropy(x, c), ag.NewScalar[T](weights[c]))
	}
}

// FocalLoss implements a variant of the CrossEntropy loss that reduces
// the loss contribution from "easy" examples and increases the importance
// of correcting misclassified examples.
// x is the raw scores for each class (logits).
// c is the index of the gold class.
// gamma is the focusing parameter (gamma ≥ 0).
func FocalLoss[T mat.DType](x ag.Node[T], c int, gamma T) ag.Node[T] {
	ce := CrossEntropy(x, c)
	p := ag.Exp(ag.Neg(ce))
	sub := ag.ReverseSub(p, ag.NewScalar[T](1.0))
	a := ag.Pow(sub, gamma)
	return ag.Prod(a, ce)
}

// WeightedFocalLoss implements a variant of the CrossEntropy loss that reduces
// the loss contribution from "easy" examples and increases the importance
// of correcting misclassified examples.
// x is the raw scores for each class (logits).
// c is the index of the gold class.
// gamma is the focusing parameter (gamma ≥ 0).
// This function is scaled by a weighting factor weights[class] ∈ [0,1].
func WeightedFocalLoss[T mat.DType](weights []T) func(x ag.Node[T], c int, gamma T) ag.Node[T] {
	return func(x ag.Node[T], c int, gamma T) ag.Node[T] {
		ce := CrossEntropy(x, c)
		p := ag.Exp(ag.Neg(ce))
		sub := ag.ReverseSub(p, ag.NewScalar[T](1.0))
		b := ag.Pow(sub, gamma)
		fl := ag.Prod(b, ce)
		return ag.ProdScalar(fl, ag.NewScalar[T](weights[c]))
	}
}

// Perplexity computes the perplexity, implemented as exp over the cross-entropy.
func Perplexity[T mat.DType](x ag.Node[T], c int) ag.Node[T] {
	return ag.Exp(CrossEntropy(x, c))
}

// ZeroOneQuantization is a loss function that is minimized when each component
// of x satisfies x(i) ≡ [x]i ∈ {0, 1}.
func ZeroOneQuantization[T mat.DType](x ag.Node[T]) ag.Node[T] {
	return ag.ReduceSum(ag.Prod(ag.Square(x), ag.Square(ag.ReverseSub(x, ag.NewScalar[T](1.0)))))
}

// Norm2Quantization is a loss function that is minimized when norm2(x) = 1.
func Norm2Quantization[T mat.DType](x ag.Node[T]) ag.Node[T] {
	return ag.Square(ag.SubScalar(ag.ReduceSum(ag.Square(x)), ag.NewScalar[T](1.0)))
}

// OneHotQuantization is a loss function that pushes towards the x vector to be 1-hot.
// q is the quantization regularizer weight (suggested  0.00001).
func OneHotQuantization[T mat.DType](x ag.Node[T], q T) ag.Node[T] {
	return ag.ProdScalar(ag.Add(ZeroOneQuantization(x), Norm2Quantization(x)), ag.NewScalar[T](q))
}

// Distance is a loss function that calculates the distance between target and x.
func Distance[T mat.DType](x ag.Node[T], target T) ag.Node[T] {
	return ag.Abs(ag.Sub(ag.NewScalar[T](target), x))
}

// MSESeq calculates the MSE loss on the given sequence.
func MSESeq[T mat.DType](predicted []ag.Node[T], target []ag.Node[T], reduceMean bool) ag.Node[T] {
	loss := MSE(predicted[0], target[0], false)
	for i := 1; i < len(predicted); i++ {
		loss = ag.Add(loss, MSE(predicted[i], target[i], false))
	}
	if reduceMean {
		return ag.DivScalar(loss, ag.NewScalar[T](T(len(predicted))))
	}
	return loss
}

// MAESeq calculates the MAE loss on the given sequence.
func MAESeq[T mat.DType](predicted []ag.Node[T], target []ag.Node[T], reduceMean bool) ag.Node[T] {
	loss := MAE(predicted[0], target[0], false)
	for i := 1; i < len(predicted); i++ {
		loss = ag.Add(loss, MAE(predicted[i], target[i], false))
	}
	if reduceMean {
		return ag.DivScalar(loss, ag.NewScalar[T](T(len(predicted))))
	}
	return loss
}

// CrossEntropySeq calculates the CrossEntropy loss on the given sequence.
func CrossEntropySeq[T mat.DType](predicted []ag.Node[T], target []int, reduceMean bool) ag.Node[T] {
	loss := CrossEntropy(predicted[0], target[0])
	for i := 1; i < len(predicted); i++ {
		loss = ag.Add(loss, CrossEntropy(predicted[i], target[i]))
	}
	if reduceMean {
		return ag.DivScalar(loss, ag.NewScalar[T](T(len(predicted))))
	}
	return loss
}

// SPG (Softmax Policy Gradient) is a Gradient Policy used in Reinforcement Learning.
// logPropActions are the log-probability of the chosen action by the Agent at each time;
// logProbTargets are results of the reward function i.e. the predicted log-likelihood of the ground truth at each time;
func SPG[T mat.DType](logPropActions []ag.Node[T], logProbTargets []ag.Node[T]) ag.Node[T] {
	var loss ag.Node[T]
	for t := 0; t < len(logPropActions); t++ {
		loss = ag.Add(loss, ag.Prod(logPropActions[t], logProbTargets[t]))
	}
	return ag.Neg(loss)
}
