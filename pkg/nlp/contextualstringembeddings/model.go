// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package contextualstringembeddings provides an implementation of the "Contextual String Embeddings"
// of words (Akbik et al., 2018).
// https://www.aclweb.org/anthology/C18-1139/
package contextualstringembeddings

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/charlm"
	"github.com/nlpodyssey/spago/pkg/utils"
	"strings"
	"sync"
)

var (
	_ nn.Model[float32] = &Model[float32]{}
)

// MergeType is the enumeration-like type used for the set of merging methods
// which a Contextual String Embeddings model Processor can perform.
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

// Model contains the serializable parameters for a Contextual String Embeddings model.
type Model[T mat.DType] struct {
	nn.BaseModel[T]
	LeftToRight *charlm.Model[T]
	RightToLeft *charlm.Model[T]
	MergeMode   MergeType
	StartMarker rune
	EndMarker   rune
}

func init() {
	gob.Register(&Model[float32]{})
	gob.Register(&Model[float64]{})
}

// New returns a new Contextual String Embeddings Model.
func New[T mat.DType](leftToRight, rightToLeft *charlm.Model[T], merge MergeType, startMarker, endMarker rune) *Model[T] {
	return &Model[T]{
		LeftToRight: leftToRight,
		RightToLeft: rightToLeft,
		MergeMode:   merge,
		StartMarker: startMarker,
		EndMarker:   endMarker,
	}
}

type wordBoundary struct {
	// index of the end of a word in the left-to-right sequence
	endIndex int
	// index of the end of a word in the right-to-left sequence
	reverseEndIndex int
}

// Encode performs the forward step for each input and returns the result.
func (m *Model[T]) Encode(words []string) []ag.Node[T] {
	text := strings.Join(words, " ")
	boundaries := makeWordBoundaries(words, text)
	sequence := utils.SplitByRune(text)

	var hiddenStates []ag.Node[T]
	var reverseHiddenStates []ag.Node[T]
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		hiddenStates = process(m.LeftToRight, padding(sequence, m.StartMarker, m.EndMarker))
	}()
	go func() {
		defer wg.Done()
		reverseHiddenStates = process(m.RightToLeft, padding(reversed(sequence), m.StartMarker, m.EndMarker))
	}()
	wg.Wait()

	out := make([]ag.Node[T], len(words))
	for i, boundary := range boundaries {
		out[i] = m.merge(reverseHiddenStates[boundary.reverseEndIndex], hiddenStates[boundary.endIndex])
	}
	return out
}

func makeWordBoundaries(words []string, text string) []wordBoundary {
	textLength := len([]rune(text)) // note the conversion to []rune
	boundaries := make([]wordBoundary, len(words))
	start := 0
	for i, word := range words {
		wordLength := len([]rune(word)) // note the conversion to []rune
		boundaries[i] = wordBoundary{
			endIndex:        start + wordLength + 1, // include startMask offset
			reverseEndIndex: textLength - start + 1, // include startMask offset
		}
		start += wordLength + 1 // include the space separator
	}
	return boundaries
}

func reversed(words []string) []string {
	r := make([]string, len(words))
	copy(r, words)
	for i := 0; i < len(r)/2; i++ {
		j := len(r) - i - 1
		r[i], r[j] = r[j], r[i]
	}
	return r
}

func padding(sequence []string, startMarker, endMarker rune) []string {
	length := len(sequence) + 2
	padded := make([]string, length)
	padded[0] = string(startMarker)
	padded[length-1] = string(endMarker)
	copy(padded[1:length-1], sequence)
	return padded
}

func process[T mat.DType](proc *charlm.Model[T], sequence []string) []ag.Node[T] {
	return proc.UseProjection(proc.RNN.Forward(proc.GetEmbeddings(sequence)...))
}

func (m *Model[T]) merge(a, b ag.Node[T]) ag.Node[T] {
	g := m.Graph()
	switch m.MergeMode {
	case Concat:
		return g.Concat(a, b)
	case Sum:
		return g.Add(a, b)
	case Prod:
		return g.Prod(a, b)
	case Avg:
		return g.ProdScalar(g.Add(a, b), g.NewScalar(0.5))
	default:
		panic("contextual string embeddings: invalid merge mode for the ContextualStringEmbeddings")
	}
}
