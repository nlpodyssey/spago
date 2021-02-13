// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package server

import (
	"github.com/nlpodyssey/spago/pkg/mat32/floatutils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/head/sequenceclassification"
	"runtime"
	"sort"
	"strconv"
	"time"
)

func (s *Server) classify(text string, text2 string) *ClassifyResponse {
	start := time.Now()

	g := ag.NewGraph(ag.IncrementalForward(false), ag.ConcurrentComputations(runtime.NumCPU()))
	defer g.Clear()
	proc := nn.Reify(nn.Context{Graph: g, Mode: nn.Inference}, s.model).(*sequenceclassification.Model)
	inputIds := getInputIDs(s.bpeTokenizer, text, text2)
	logits := proc.Classify(inputIds)
	g.Forward()

	probs := floatutils.SoftMax(g.GetCopiedValue(logits).Data())
	best := floatutils.ArgMax(probs)
	classes := s.model.(*sequenceclassification.Model).BART.Config.ID2Label
	class := classes[strconv.Itoa(best)]

	distribution := make([]ClassConfidencePair, len(probs))
	for i := 0; i < len(probs); i++ {
		distribution[i] = ClassConfidencePair{
			Class:      classes[strconv.Itoa(i)],
			Confidence: probs[i],
		}
	}

	sort.Slice(distribution, func(i, j int) bool {
		return distribution[i].Confidence > distribution[j].Confidence
	})

	return &ClassifyResponse{
		Class:        class,
		Confidence:   probs[best],
		Distribution: distribution,
		Took:         time.Since(start).Milliseconds(),
	}
}
