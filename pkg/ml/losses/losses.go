// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package losses

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
)

// MAE measures the mean absolute error (a.k.a. L1 Loss) between each element in the input x and target y.
func MAE(g *ag.Graph, x ag.Node, y ag.Node, reduceMean bool) ag.Node {
	loss := g.Abs(g.Sub(x, y))
	if reduceMean {
		return g.ReduceMean(loss)
	}
	return g.ReduceSum(loss)
}

// MSE measures the mean squared error (squared L2 norm) between each element in the input x and target y.
func MSE(g *ag.Graph, x ag.Node, y ag.Node, reduceMean bool) ag.Node {
	loss := g.ProdScalar(g.Square(g.Sub(x, y)), g.Constant(0.5))
	if reduceMean {
		return g.ReduceMean(loss)
	}
	return g.ReduceSum(loss)
}

// NLL returns the loss of the input x respect to the target y.
// The target is expected to be a one-hot vector.
func NLL(g *ag.Graph, x ag.Node, y ag.Node) ag.Node {
	return g.Neg(g.ReduceSum(g.Prod(y, g.Log(x))))
}

// CrossEntropy implements a cross-entropy loss function.
// x is the raw scores for each class (logits).
// c is the index of the gold class.
func CrossEntropy(g *ag.Graph, x ag.Node, c int) ag.Node {
	return g.Add(g.Neg(g.AtVec(x, c)), g.Log(g.ReduceSum(g.Exp(x))))
}

// WeightedCrossEntropy implements a weighted cross-entropy loss function.
// x is the raw scores for each class (logits).
// c is the index of the gold class.
// This function is scaled by a weighting factor weights[class] ∈ [0,1]
func WeightedCrossEntropy(weights []mat.Float) func(g *ag.Graph, x ag.Node, c int) ag.Node {
	return func(g *ag.Graph, x ag.Node, c int) ag.Node {
		return g.ProdScalar(CrossEntropy(g, x, c), g.NewScalar(weights[c]))
	}
}

// FocalLoss implements a variant of the CrossEntropy loss that reduces
// the loss contribution from "easy" examples and increases the importance
// of correcting misclassified examples.
// x is the raw scores for each class (logits).
// c is the index of the gold class.
// gamma is the focusing parameter (gamma ≥ 0).
func FocalLoss(g *ag.Graph, x ag.Node, c int, gamma mat.Float) ag.Node {
	ce := CrossEntropy(g, x, c)
	p := g.Exp(g.Neg(ce))
	sub := g.ReverseSub(p, g.NewScalar(1.0))
	a := g.Pow(sub, gamma)
	return g.Prod(a, ce)
}

// WeightedFocalLoss implements a variant of the CrossEntropy loss that reduces
// the loss contribution from "easy" examples and increases the importance
// of correcting misclassified examples.
// x is the raw scores for each class (logits).
// c is the index of the gold class.
// gamma is the focusing parameter (gamma ≥ 0).
// This function is scaled by a weighting factor weights[class] ∈ [0,1].
func WeightedFocalLoss(weights []mat.Float) func(g *ag.Graph, x ag.Node, c int, gamma mat.Float) ag.Node {
	return func(g *ag.Graph, x ag.Node, c int, gamma mat.Float) ag.Node {
		ce := CrossEntropy(g, x, c)
		p := g.Exp(g.Neg(ce))
		sub := g.ReverseSub(p, g.NewScalar(1.0))
		b := g.Pow(sub, gamma)
		fl := g.Prod(b, ce)
		return g.ProdScalar(fl, g.NewScalar(weights[c]))
	}
}

// Perplexity computes the perplexity, implemented as exp over the cross-entropy.
func Perplexity(g *ag.Graph, x ag.Node, c int) ag.Node {
	return g.Exp(CrossEntropy(g, x, c))
}

// ZeroOneQuantization is a loss function that is minimized when each component
// of x satisfies x(i) ≡ [x]i ∈ {0, 1}.
func ZeroOneQuantization(g *ag.Graph, x ag.Node) ag.Node {
	return g.ReduceSum(g.Prod(g.Square(x), g.Square(g.ReverseSub(x, g.NewScalar(1.0)))))
}

// Norm2Quantization is a loss function that is minimized when norm2(x) = 1.
func Norm2Quantization(g *ag.Graph, x ag.Node) ag.Node {
	return g.Square(g.SubScalar(g.ReduceSum(g.Square(x)), g.NewScalar(1.0)))
}

// OneHotQuantization is a loss function that pushes towards the x vector to be 1-hot.
// q is the quantization regularizer weight (suggested  0.00001).
func OneHotQuantization(g *ag.Graph, x ag.Node, q mat.Float) ag.Node {
	return g.ProdScalar(g.Add(ZeroOneQuantization(g, x), Norm2Quantization(g, x)), g.NewScalar(q))
}

// Distance is a loss function that calculates the distance between target and x.
func Distance(g *ag.Graph, x ag.Node, target mat.Float) ag.Node {
	return g.Abs(g.Sub(g.NewScalar(target), x))
}

// MSESeq calculates the MSE loss on the given sequence.
func MSESeq(g *ag.Graph, predicted []ag.Node, target []ag.Node, reduceMean bool) ag.Node {
	loss := MSE(g, predicted[0], target[0], false)
	for i := 1; i < len(predicted); i++ {
		loss = g.Add(loss, MSE(g, predicted[i], target[i], false))
	}
	if reduceMean {
		return g.DivScalar(loss, g.NewScalar(mat.Float(len(predicted))))
	}
	return loss
}

// MAESeq calculates the MAE loss on the given sequence.
func MAESeq(g *ag.Graph, predicted []ag.Node, target []ag.Node, reduceMean bool) ag.Node {
	loss := MAE(g, predicted[0], target[0], false)
	for i := 1; i < len(predicted); i++ {
		loss = g.Add(loss, MAE(g, predicted[i], target[i], false))
	}
	if reduceMean {
		return g.DivScalar(loss, g.NewScalar(mat.Float(len(predicted))))
	}
	return loss
}

// CrossEntropySeq calculates the CrossEntropy loss on the given sequence.
func CrossEntropySeq(g *ag.Graph, predicted []ag.Node, target []int, reduceMean bool) ag.Node {
	loss := CrossEntropy(g, predicted[0], target[0])
	for i := 1; i < len(predicted); i++ {
		loss = g.Add(loss, CrossEntropy(g, predicted[i], target[i]))
	}
	if reduceMean {
		return g.DivScalar(loss, g.NewScalar(mat.Float(len(predicted))))
	}
	return loss
}

// SPG (Softmax Policy Gradient) is a Gradient Policy used in Reinforcement Learning.
// logPropActions are the log-probability of the chosen action by the Agent at each time;
// logProbTargets are results of the reward function i.e. the predicted log-likelihood of the ground truth at each time;
func SPG(g *ag.Graph, logPropActions []ag.Node, logProbTargets []ag.Node) ag.Node {
	var loss ag.Node
	for t := 0; t < len(logPropActions); t++ {
		loss = g.Add(loss, g.Prod(logPropActions[t], logProbTargets[t]))
	}
	return g.Neg(loss)
}
