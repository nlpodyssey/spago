// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package losses

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

// MAE measures the mean absolute error (a.k.a. L1 Loss) between each element in the input x and target y.
func MAE(x mat.Tensor, y mat.Tensor, reduceMean bool) mat.Tensor {
	loss := ag.Abs(ag.Sub(x, y))
	if reduceMean {
		return ag.ReduceMean(loss)
	}
	return ag.ReduceSum(loss)
}

// MSE measures the mean squared error (squared L2 norm) between each element in the input x and target y.
func MSE(x mat.Tensor, y mat.Tensor, reduceMean bool) mat.Tensor {
	loss := ag.ProdScalar(ag.Square(ag.Sub(x, y)), x.Value().(mat.Matrix).NewScalar(0.5))
	if reduceMean {
		return ag.ReduceMean(loss)
	}
	return ag.ReduceSum(loss)
}

// NLL returns the loss of the input x respect to the target y.
// The target is expected to be a one-hot vector.
func NLL(x mat.Tensor, y mat.Tensor) mat.Tensor {
	return ag.Neg(ag.ReduceSum(ag.Prod(y, ag.Log(x))))
}

// CrossEntropy implements a cross-entropy loss function.
// x is the raw scores for each class (logits).
// c is the index of the gold class.
func CrossEntropy(x mat.Tensor, c int) mat.Tensor {
	return ag.Add(ag.Neg(ag.At(x, c)), ag.LogSumExp(x))
}

// WeightedCrossEntropy implements a weighted cross-entropy loss function.
// x is the raw scores for each class (logits).
// c is the index of the gold class.
// This function is scaled by a weighting factor weights[class] ∈ [0,1]
func WeightedCrossEntropy(weights mat.Matrix) func(x mat.Tensor, c int) mat.Tensor {
	return func(x mat.Tensor, c int) mat.Tensor {
		return ag.ProdScalar(CrossEntropy(x, c), weights.At(c))
	}
}

// FocalLoss implements a variant of the CrossEntropy loss that reduces
// the loss contribution from "easy" examples and increases the importance
// of correcting misclassified examples.
// x is the raw scores for each class (logits).
// c is the index of the gold class.
// gamma is the focusing parameter (gamma ≥ 0).
func FocalLoss(x mat.Tensor, c int, gamma float64) mat.Tensor {
	ce := CrossEntropy(x, c)
	p := ag.Exp(ag.Neg(ce))
	sub := ag.ReverseSub(p, x.Value().(mat.Matrix).NewScalar(1))
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
func WeightedFocalLoss(weights mat.Matrix) func(x mat.Tensor, c int, gamma float64) mat.Tensor {
	return func(x mat.Tensor, c int, gamma float64) mat.Tensor {
		ce := CrossEntropy(x, c)
		p := ag.Exp(ag.Neg(ce))
		sub := ag.ReverseSub(p, x.Value().(mat.Matrix).NewScalar(1.0))
		b := ag.Pow(sub, gamma)
		fl := ag.Prod(b, ce)
		return ag.ProdScalar(fl, weights.At(c))
	}
}

// Perplexity computes the perplexity, implemented as exp over the cross-entropy.
func Perplexity(x mat.Tensor, c int) mat.Tensor {
	return ag.Exp(CrossEntropy(x, c))
}

// ZeroOneQuantization is a loss function that is minimized when each component
// of x satisfies x(i) ≡ [x]i ∈ {0, 1}.
func ZeroOneQuantization(x mat.Tensor) mat.Tensor {
	return ag.ReduceSum(ag.Prod(ag.Square(x), ag.Square(ag.ReverseSub(x, x.Value().(mat.Matrix).NewScalar(1.0)))))
}

// Norm2Quantization is a loss function that is minimized when norm2(x) = 1.
func Norm2Quantization(x mat.Tensor) mat.Tensor {
	return ag.Square(ag.SubScalar(ag.ReduceSum(ag.Square(x)), x.Value().(mat.Matrix).NewScalar(1.0)))
}

// OneHotQuantization is a loss function that pushes towards the x vector to be 1-hot.
// q is the quantization regularizer weight (suggested  0.00001).
func OneHotQuantization(x mat.Tensor, q float64) mat.Tensor {
	return ag.ProdScalar(ag.Add(ZeroOneQuantization(x), Norm2Quantization(x)), x.Value().(mat.Matrix).NewScalar(q))
}

// Distance is a loss function that calculates the distance between target and x.
func Distance(x mat.Tensor, target float64) mat.Tensor {
	return ag.Abs(ag.Sub(x.Value().(mat.Matrix).NewScalar(target), x))
}

// MSESeq calculates the MSE loss on the given sequence.
func MSESeq(predicted []mat.Tensor, target []mat.Tensor, reduceMean bool) mat.Tensor {
	loss := MSE(predicted[0], target[0], false)
	for i := 1; i < len(predicted); i++ {
		loss = ag.Add(loss, MSE(predicted[i], target[i], false))
	}
	if reduceMean {
		return ag.DivScalar(loss, loss.Value().(mat.Matrix).NewScalar(float64(len(predicted))))
	}
	return loss
}

// MAESeq calculates the MAE loss on the given sequence.
func MAESeq(predicted []mat.Tensor, target []mat.Tensor, reduceMean bool) mat.Tensor {
	loss := MAE(predicted[0], target[0], false)
	for i := 1; i < len(predicted); i++ {
		loss = ag.Add(loss, MAE(predicted[i], target[i], false))
	}
	if reduceMean {
		return ag.DivScalar(loss, loss.Value().(mat.Matrix).NewScalar(float64(len(predicted))))
	}
	return loss
}

// CrossEntropySeq calculates the CrossEntropy loss on the given sequence.
func CrossEntropySeq(predicted []mat.Tensor, target []int, reduceMean bool) mat.Tensor {
	loss := CrossEntropy(predicted[0], target[0])
	for i := 1; i < len(predicted); i++ {
		loss = ag.Add(loss, CrossEntropy(predicted[i], target[i]))
	}
	if reduceMean {
		return ag.DivScalar(loss, loss.Value().(mat.Matrix).NewScalar(float64(len(predicted))))
	}
	return loss
}

// SPG (Softmax Policy Gradient) is a Gradient Policy used in Reinforcement Learning.
// logPropActions are the log-probability of the chosen action by the Agent at each time;
// logProbTargets are results of the reward function i.e. the predicted log-likelihood of the ground truth at each time;
func SPG(logPropActions []mat.Tensor, logProbTargets []mat.Tensor) mat.Tensor {
	var loss mat.Tensor
	for t := 0; t < len(logPropActions); t++ {
		loss = ag.Add(loss, ag.Prod(logPropActions[t], logProbTargets[t]))
	}
	return ag.Neg(loss)
}

// Huber measures the Huber loss between each element in the input x and target y, controlled by
// the threshold (delta). Below the threshold, it behaves like MSE; above it, it becomes linear
// in order to reduce the effect of outliers. If reduceMean is true, it returns the average loss;
// otherwise it returns the sum of the losses.
//
// Huber(d) = { 0.5 * (d^2)              if |d| ≤ δ
//
//	δ * (|d| - 0.5 * δ)      otherwise }
//
// Here, d = x - y.
func Huber(x, y mat.Tensor, delta float64, reduceMean bool) mat.Tensor {
	// 1) Compute d = x - y, then |d|
	d := ag.Sub(x, y)
	absD := ag.Abs(d)

	// 2) Build a scalar tensor from 'delta'
	//    then multiply it by the shape of absD (via OnesLike) to broadcast
	//    the scalar across all elements. This avoids dimension mismatch with Min().
	deltaMat := x.Value().(mat.Matrix).NewScalar(delta)
	deltaVec := ag.ProdScalar(x.Value().(mat.Matrix).OnesLike(), deltaMat)

	// 3) clipped = min(|d|, deltaVec)
	clipped := ag.Min(absD, deltaVec)

	// 4) 0.5 * (clipped)^2
	halfSqr := ag.ProdScalar(ag.Square(clipped), x.Value().(mat.Matrix).NewScalar(0.5))

	// 5) deltaVec * (|d| - clipped)
	linear := ag.Prod(deltaVec, ag.Sub(absD, clipped))

	// 6) Combine
	loss := ag.Add(halfSqr, linear)

	// 7) reduceMean or sum
	if reduceMean {
		return ag.ReduceMean(loss)
	}
	return ag.ReduceSum(loss)
}

// HuberSeq calculates the Huber loss on multiple (predicted, target) pairs.
// It sums the Huber loss across the entire sequence, optionally averaging it
// by the number of elements if reduceMean is true.
func HuberSeq(predicted, target []mat.Tensor, delta float64, reduceMean bool) mat.Tensor {
	// Accumulate the Huber loss across the sequence
	loss := Huber(predicted[0], target[0], delta, false)
	for i := 1; i < len(predicted); i++ {
		loss = ag.Add(loss, Huber(predicted[i], target[i], delta, false))
	}

	// Optionally divide by length to get mean
	if reduceMean {
		return ag.DivScalar(loss, loss.Value().(mat.Matrix).NewScalar(float64(len(predicted))))
	}
	return loss
}
