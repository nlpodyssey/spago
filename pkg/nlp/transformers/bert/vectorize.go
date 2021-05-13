// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"fmt"
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/wordpiecetokenizer"
	"runtime"
)

// PoolingStrategy defines the method to obtain the dense sentence representation
type PoolingStrategy int

const (
	// ClsToken gets the encoding state corresponding to [CLS], i.e. the first token (default)
	ClsToken PoolingStrategy = iota
	// ReduceMean takes the average of the encoding states
	ReduceMean
	// ReduceMax takes the maximum of the encoding states
	ReduceMax
	// ReduceMeanMax does ReduceMean and ReduceMax separately and then concat them together
	ReduceMeanMax
)

// Vectorize transforms the text into a dense vector representation.
func (m *Model) Vectorize(text string, poolingStrategy PoolingStrategy) (mat.Matrix, error) {
	tokenizer := wordpiecetokenizer.New(m.Vocabulary)
	origTokens := tokenizer.Tokenize(text)
	tokenized := pad(tokenizers.GetStrings(origTokens))

	g := ag.NewGraph(ag.ConcurrentComputations(runtime.NumCPU()))
	defer g.Clear()
	proc := nn.ReifyForInference(m, g).(*Model)
	encoded := proc.Encode(tokenized)

	var pooled ag.Node
	switch poolingStrategy {
	case ReduceMean:
		pooled = g.Mean(encoded)
	case ReduceMax:
		pooled = max(g, encoded)
	case ReduceMeanMax:
		pooled = g.Concat(g.Mean(encoded), max(g, encoded))
	case ClsToken:
		pooled = proc.Pool(encoded)
	default:
		return nil, fmt.Errorf("bert: invalid pooling strategy")
	}

	return g.GetCopiedValue(pooled), nil
}

// Max returns the value that describes the maximum of the sample.
func max(g *ag.Graph, xs []ag.Node) ag.Node {
	maxVector := xs[0]
	for i := 1; i < len(xs); i++ {
		maxVector = g.Max(maxVector, xs[i])
	}
	return maxVector
}
