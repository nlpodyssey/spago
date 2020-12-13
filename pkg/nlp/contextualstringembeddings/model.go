// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package contextualstringembeddings provides an implementation of the "Contextual String Embeddings"
// of words (Akbik et al., 2018).
// https://www.aclweb.org/anthology/C18-1139/
package contextualstringembeddings

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/charlm"
	"github.com/nlpodyssey/spago/pkg/utils"
	"strings"
	"sync"
)

var (
	_ nn.Model     = &Model{}
	_ nn.Processor = &Processor{}
)

type MergeType int

const (
	Concat MergeType = iota // The outputs are concatenated together (the default)
	Sum                     // The outputs are added together
	Prod                    // The outputs are multiplied element-wise together
	Avg                     // The average of the outputs is taken
)

type Model struct {
	LeftToRight *charlm.Model
	RightToLeft *charlm.Model
	MergeMode   MergeType
	StartMarker rune
	EndMarker   rune
}

func New(leftToRight, rightToLeft *charlm.Model, merge MergeType, startMarker, endMarker rune) *Model {
	return &Model{
		LeftToRight: leftToRight,
		RightToLeft: rightToLeft,
		MergeMode:   merge,
		StartMarker: startMarker,
		EndMarker:   endMarker,
	}
}

type Processor struct {
	nn.BaseProcessor
	mergeMode   MergeType
	leftToRight *charlm.Processor
	rightToLeft *charlm.Processor
}

func (m *Model) NewProc(ctx nn.Context) nn.Processor {
	return &Processor{
		BaseProcessor: nn.BaseProcessor{
			Model:             m,
			Mode:              ctx.Mode,
			Graph:             ctx.Graph,
			FullSeqProcessing: true,
		},
		mergeMode:   m.MergeMode,
		leftToRight: m.LeftToRight.NewProc(ctx).(*charlm.Processor),
		rightToLeft: m.RightToLeft.NewProc(ctx).(*charlm.Processor),
	}
}

type wordBoundary struct {
	// index of the end of a word in the left-to-right sequence
	endIndex int
	// index of the end of a word in the right-to-left sequence
	reverseEndIndex int
}

func (p Processor) Encode(words []string) []ag.Node {
	text := strings.Join(words, " ")
	boundaries := makeWordBoundaries(words, text)
	sequence := utils.SplitByRune(text)

	var hiddenStates []ag.Node
	var reverseHiddenStates []ag.Node
	var wg sync.WaitGroup
	wg.Add(2)
	m := p.Model.(*Model)
	go func() {
		defer wg.Done()
		hiddenStates = process(p.leftToRight, padding(sequence, m.StartMarker, m.EndMarker))
	}()
	go func() {
		defer wg.Done()
		reverseHiddenStates = process(p.rightToLeft, padding(reversed(sequence), m.StartMarker, m.EndMarker))
	}()
	wg.Wait()

	out := make([]ag.Node, len(words))
	for i, boundary := range boundaries {
		out[i] = p.merge(reverseHiddenStates[boundary.reverseEndIndex], hiddenStates[boundary.endIndex])
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

func process(proc *charlm.Processor, sequence []string) []ag.Node {
	return proc.UseProjection(proc.RNN.Forward(proc.GetEmbeddings(sequence)...)...)
}

func (p *Processor) merge(a, b ag.Node) ag.Node {
	g := p.Graph
	switch p.mergeMode {
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

func (p Processor) Forward(_ ...ag.Node) []ag.Node {
	panic("contextual string embeddings: method not implemented. Use Encode() instead.")
}
