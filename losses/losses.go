// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package losses

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

// MAE measures the mean absolute error (a.k.a. L1 Loss) between each element in the input x and target y.
func MAE(x ag.DualValue, y ag.DualValue, reduceMean bool) ag.DualValue {
	loss := ag.Abs(ag.Sub(x, y))
	if reduceMean {
		return ag.ReduceMean(loss)
	}
	return ag.ReduceSum(loss)
}

// MSE measures the mean squared error (squared L2 norm) between each element in the input x and target y.
func MSE(x ag.DualValue, y ag.DualValue, reduceMean bool) ag.DualValue {
	loss := ag.ProdScalar(ag.Square(ag.Sub(x, y)), x.Value().NewScalar(0.5))
	if reduceMean {
		return ag.ReduceMean(loss)
	}
	return ag.ReduceSum(loss)
}

// NLL returns the loss of the input x respect to the target y.
// The target is expected to be a one-hot vector.
func NLL(x ag.DualValue, y ag.DualValue) ag.DualValue {
	return ag.Neg(ag.ReduceSum(ag.Prod(y, ag.Log(x))))
}

// CrossEntropy implements a cross-entropy loss function.
// x is the raw scores for each class (logits).
// c is the index of the gold class.
func CrossEntropy(x ag.DualValue, c int) ag.DualValue {
	return ag.Add(ag.Neg(ag.At(x, c)), ag.LogSumExp(x))
}

// WeightedCrossEntropy implements a weighted cross-entropy loss function.
// x is the raw scores for each class (logits).
// c is the index of the gold class.
// This function is scaled by a weighting factor weights[class] ∈ [0,1]
func WeightedCrossEntropy(weights mat.Matrix) func(x ag.DualValue, c int) ag.DualValue {
	return func(x ag.DualValue, c int) ag.DualValue {
		return ag.ProdScalar(CrossEntropy(x, c), weights.At(c))
	}
}

// FocalLoss implements a variant of the CrossEntropy loss that reduces
// the loss contribution from "easy" examples and increases the importance
// of correcting misclassified examples.
// x is the raw scores for each class (logits).
// c is the index of the gold class.
// gamma is the focusing parameter (gamma ≥ 0).
func FocalLoss(x ag.DualValue, c int, gamma float64) ag.DualValue {
	ce := CrossEntropy(x, c)
	p := ag.Exp(ag.Neg(ce))
	sub := ag.ReverseSub(p, x.Value().NewScalar(1))
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
func WeightedFocalLoss(weights mat.Matrix) func(x ag.DualValue, c int, gamma float64) ag.DualValue {
	return func(x ag.DualValue, c int, gamma float64) ag.DualValue {
		ce := CrossEntropy(x, c)
		p := ag.Exp(ag.Neg(ce))
		sub := ag.ReverseSub(p, x.Value().NewScalar(1.0))
		b := ag.Pow(sub, gamma)
		fl := ag.Prod(b, ce)
		return ag.ProdScalar(fl, weights.At(c))
	}
}

// Perplexity computes the perplexity, implemented as exp over the cross-entropy.
func Perplexity(x ag.DualValue, c int) ag.DualValue {
	return ag.Exp(CrossEntropy(x, c))
}

// ZeroOneQuantization is a loss function that is minimized when each component
// of x satisfies x(i) ≡ [x]i ∈ {0, 1}.
func ZeroOneQuantization(x ag.DualValue) ag.DualValue {
	return ag.ReduceSum(ag.Prod(ag.Square(x), ag.Square(ag.ReverseSub(x, x.Value().NewScalar(1.0)))))
}

// Norm2Quantization is a loss function that is minimized when norm2(x) = 1.
func Norm2Quantization(x ag.DualValue) ag.DualValue {
	return ag.Square(ag.SubScalar(ag.ReduceSum(ag.Square(x)), x.Value().NewScalar(1.0)))
}

// OneHotQuantization is a loss function that pushes towards the x vector to be 1-hot.
// q is the quantization regularizer weight (suggested  0.00001).
func OneHotQuantization(x ag.DualValue, q float64) ag.DualValue {
	return ag.ProdScalar(ag.Add(ZeroOneQuantization(x), Norm2Quantization(x)), x.Value().NewScalar(q))
}

// Distance is a loss function that calculates the distance between target and x.
func Distance(x ag.DualValue, target float64) ag.DualValue {
	return ag.Abs(ag.Sub(x.Value().NewScalar(target), x))
}

// MSESeq calculates the MSE loss on the given sequence.
func MSESeq(predicted []ag.DualValue, target []ag.DualValue, reduceMean bool) ag.DualValue {
	loss := MSE(predicted[0], target[0], false)
	for i := 1; i < len(predicted); i++ {
		loss = ag.Add(loss, MSE(predicted[i], target[i], false))
	}
	if reduceMean {
		return ag.DivScalar(loss, loss.Value().NewScalar(float64(len(predicted))))
	}
	return loss
}

// MAESeq calculates the MAE loss on the given sequence.
func MAESeq(predicted []ag.DualValue, target []ag.DualValue, reduceMean bool) ag.DualValue {
	loss := MAE(predicted[0], target[0], false)
	for i := 1; i < len(predicted); i++ {
		loss = ag.Add(loss, MAE(predicted[i], target[i], false))
	}
	if reduceMean {
		return ag.DivScalar(loss, loss.Value().NewScalar(float64(len(predicted))))
	}
	return loss
}

// CrossEntropySeq calculates the CrossEntropy loss on the given sequence.
func CrossEntropySeq(predicted []ag.DualValue, target []int, reduceMean bool) ag.DualValue {
	loss := CrossEntropy(predicted[0], target[0])
	for i := 1; i < len(predicted); i++ {
		loss = ag.Add(loss, CrossEntropy(predicted[i], target[i]))
	}
	if reduceMean {
		return ag.DivScalar(loss, loss.Value().NewScalar(float64(len(predicted))))
	}
	return loss
}

// SPG (Softmax Policy Gradient) is a Gradient Policy used in Reinforcement Learning.
// logPropActions are the log-probability of the chosen action by the Agent at each time;
// logProbTargets are results of the reward function i.e. the predicted log-likelihood of the ground truth at each time;
func SPG(logPropActions []ag.DualValue, logProbTargets []ag.DualValue) ag.DualValue {
	var loss ag.DualValue
	for t := 0; t < len(logPropActions); t++ {
		loss = ag.Add(loss, ag.Prod(logPropActions[t], logProbTargets[t]))
	}
	return ag.Neg(loss)
}
