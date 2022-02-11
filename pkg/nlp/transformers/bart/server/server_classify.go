// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package server

import (
	"github.com/nlpodyssey/spago/pkg/mat/matutils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/head/sequenceclassification"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/tasks"
	"runtime"
	"sort"
	"strconv"
	"time"
)

func (s *Server[T]) classify(text string, text2 string) *tasks.ClassifyResponse[T] {
	start := time.Now()

	g := ag.NewGraph[T](ag.IncrementalForward[T](false), ag.ConcurrentComputations[T](runtime.NumCPU()))
	defer g.Clear()
	proc := nn.ReifyForInference(s.model, g).(*sequenceclassification.Model[T])
	inputIds := getInputIDs(s.bpeTokenizer, text, text2)
	logits := proc.Classify(inputIds)
	g.Forward()

	probs := matutils.SoftMax(g.GetCopiedValue(logits).Data())
	best := matutils.ArgMax(probs)
	classes := s.model.(*sequenceclassification.Model[T]).BART.Config.ID2Label
	class := classes[strconv.Itoa(best)]

	distribution := make([]tasks.ClassConfidencePair[T], len(probs))
	for i := 0; i < len(probs); i++ {
		distribution[i] = tasks.ClassConfidencePair[T]{
			Class:      classes[strconv.Itoa(i)],
			Confidence: probs[i],
		}
	}

	sort.Slice(distribution, func(i, j int) bool {
		return distribution[i].Confidence > distribution[j].Confidence
	})

	return &tasks.ClassifyResponse[T]{
		Class:        class,
		Confidence:   probs[best],
		Distribution: distribution,
		Took:         time.Since(start).Milliseconds(),
	}
}
