// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package attention

import (
	"saientist.dev/spago/pkg/mat"
	"saientist.dev/spago/pkg/ml/ag"
	"sync"
)

// ScaledDotProductAttention is a self-attention mechanism relating different positions of a single sequence in order to compute a representation of the same sequence.
// This method requires that the query, the key and the value vectors have already been obtained from the input sequence.
// The scaled factor is the square root of the dimension of the key vectors.
func ScaledDotProductAttention(g *ag.Graph, qs, ks, vs []ag.Node, scaledFactor float64) (context []ag.Node, probs []mat.Matrix) {
	context = make([]ag.Node, len(qs))
	probs = make([]mat.Matrix, len(qs))
	keys := g.Stack(ks...)
	values := g.T(g.Stack(vs...))
	divTerm := g.NewScalar(scaledFactor)
	for i, q := range qs {
		attScores := g.DivScalar(g.Mul(keys, q), divTerm)
		attProbs := g.Softmax(attScores)
		context[i] = g.Mul(values, attProbs)
		probs[i] = attProbs.Value()
	}
	return
}

// ScaledDotProductAttentionConcurrent does the same thing as ScaledDotProductAttention but processes input concurrently.
func ScaledDotProductAttentionConcurrent(g *ag.Graph, qs, ks, vs []ag.Node, scaledFactor float64) (context []ag.Node, probs []mat.Matrix) {
	context = make([]ag.Node, len(qs))
	probs = make([]mat.Matrix, len(qs))
	keys := g.Stack(ks...)
	values := g.T(g.Stack(vs...))
	divTerm := g.NewScalar(scaledFactor)
	var wg sync.WaitGroup
	wg.Add(len(qs))
	for i, q := range qs {
		go func(i int, q ag.Node) {
			defer wg.Done()
			attScores := g.DivScalar(g.Mul(keys, q), divTerm)
			attProbs := g.Softmax(attScores)
			context[i] = g.Mul(values, attProbs)
			probs[i] = attProbs.Value()
		}(i, q)
	}
	wg.Wait()
	return
}
