// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package attention

import (
	"saientist.dev/spago/pkg/ml/ag"
	"sync"
)

// SelfAttention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the same sequence.
// This method requires that the query, the key and the value vectors have already been obtained from the input sequence.
// The scaled factor is the square root of the dimension of the key vectors.
func SelfAttention(g *ag.Graph, qs, ks, vs []ag.Node, scaledFactor float64) []ag.Node {
	ys := make([]ag.Node, len(qs))
	keys := g.Stack(ks...)
	values := g.T(g.Stack(vs...))
	divTerm := g.NewScalar(scaledFactor)
	for i, q := range qs {
		scores := g.Softmax(g.DivScalar(g.Mul(keys, q), divTerm))
		ys[i] = g.Mul(values, scores)
	}
	return ys
}

// SelfAttentionConcurrent does the same thing as SelfAttention but processes input concurrently.
func SelfAttentionConcurrent(g *ag.Graph, qs, ks, vs []ag.Node, scaledFactor float64) []ag.Node {
	ys := make([]ag.Node, len(qs))
	keys := g.Stack(ks...)
	values := g.T(g.Stack(vs...))
	divTerm := g.NewScalar(scaledFactor)
	var wg sync.WaitGroup
	wg.Add(len(qs))
	for i, q := range qs {
		go func(i int, q ag.Node) {
			defer wg.Done()
			scores := g.Softmax(g.DivScalar(g.Mul(keys, q), divTerm))
			ys[i] = g.Mul(values, scores)
		}(i, q)
	}
	wg.Wait()
	return ys
}
