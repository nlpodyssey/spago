// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package birnn

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"sync"
)

// MergeType is the enumeration-like type used for the set of merging methods
// which a BiRNN model Processor can perform.
type MergeType int

const (
	// Concat merging method: the outputs are concatenated together (the default)
	Concat MergeType = iota
	// Sum merging method: the outputs are added together
	Sum
	// Prod merging method: the outputs multiplied element-wise together
	Prod
	// Avg merging method: the average of the outputs is taken
	Avg
)

var _ nn.Model = &Model[float32]{}

// Model contains the serializable parameters.
type Model[T mat.DType] struct {
	nn.BaseModel
	Positive  nn.StandardModel[T] // positive time direction a.k.a. left-to-right
	Negative  nn.StandardModel[T] // negative time direction a.k.a. right-to-left
	MergeMode MergeType
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new model with parameters initialized to zeros.
func New[T mat.DType](positive, negative nn.StandardModel[T], merge MergeType) *Model[T] {
	return &Model[T]{
		Positive:  positive,
		Negative:  negative,
		MergeMode: merge,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *Model[T]) Forward(xs ...ag.Node[T]) []ag.Node[T] {
	var pos []ag.Node[T]
	var neg []ag.Node[T]
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		pos = m.Positive.Forward(xs...)
	}()
	go func() {
		defer wg.Done()
		neg = m.Negative.Forward(reversed(xs)...)
	}()
	wg.Wait()
	out := make([]ag.Node[T], len(pos))
	for i := range out {
		out[i] = m.merge(pos[i], neg[len(out)-1-i])
	}
	return out
}

func reversed[T mat.DType](ns []ag.Node[T]) []ag.Node[T] {
	r := make([]ag.Node[T], len(ns))
	copy(r, ns)
	for i := 0; i < len(r)/2; i++ {
		j := len(r) - i - 1
		r[i], r[j] = r[j], r[i]
	}
	return r
}

func (m *Model[T]) merge(a, b ag.Node[T]) ag.Node[T] {
	switch m.MergeMode {
	case Concat:
		return ag.Concat(a, b)
	case Sum:
		return ag.Add(a, b)
	case Prod:
		return ag.Prod(a, b)
	case Avg:
		return ag.ProdScalar(ag.Add(a, b), a.Graph().Constant(0.5))
	default:
		panic("birnn: invalid merge mode")
	}
}
