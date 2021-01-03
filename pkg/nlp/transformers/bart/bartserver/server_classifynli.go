// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bartserver

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/floatutils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/bpetokenizer"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/barthead"
	"github.com/nlpodyssey/spago/pkg/utils/workerpool"
	"github.com/pkg/errors"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
)

type premiseHypothesisPair struct {
	index      int
	premise    string
	hypothesis string
}

const defaultHypothesisTemplate = "This text is about {}."

func (s *ServerForSequenceClassification) classifyNLI(
	text string,
	hypothesisTemplate string,
	candidateLabels []string,
	multiClass bool,
) (*ClassifyResponse, error) {
	start := time.Now()

	if hypothesisTemplate == "" {
		hypothesisTemplate = defaultHypothesisTemplate
	}

	entailmentID, contradictionID, err := s.getEntailmentAndContradictionIDs()
	if err != nil {
		return nil, err
	}

	numOfCandidateLabels := len(candidateLabels)
	logits := make([]*mat.Dense, numOfCandidateLabels)

	numWorkers := runtime.NumCPU() / 2 // leave some space for other concurrent computations
	wp := workerpool.New(numWorkers)
	workers := s.newWorkers(numWorkers)
	wg := sync.WaitGroup{}
	go wp.Run(func(workerID int, jobData interface{}) {
		data := jobData.(premiseHypothesisPair)
		logits[data.index] = workers[workerID].process(data)
		wg.Done()
	})

	for i, label := range candidateLabels {
		wg.Add(1)
		wp.PublishJobData(premiseHypothesisPair{
			index:      i,
			premise:    text,
			hypothesis: strings.Replace(hypothesisTemplate, "{}", label, -1),
		})
	}
	wg.Wait()

	if numOfCandidateLabels == 1 {
		multiClass = true
	}

	scores := func() []mat.Float {
		if multiClass {
			return getMultiClassScores(logits, entailmentID, contradictionID)
		}
		return getScores(logits, entailmentID)
	}()

	best := floatutils.ArgMax(scores)
	class := candidateLabels[best]

	distribution := make([]ClassConfidencePair, len(scores))
	for i := 0; i < len(scores); i++ {
		distribution[i] = ClassConfidencePair{
			Class:      candidateLabels[i],
			Confidence: scores[i],
		}
	}

	sort.Slice(distribution, func(i, j int) bool {
		return distribution[i].Confidence > distribution[j].Confidence
	})

	return &ClassifyResponse{
		Class:        class,
		Confidence:   scores[best],
		Distribution: distribution,
		Took:         time.Since(start).Milliseconds(),
	}, nil
}

// getMultiClassScores softmax over the entailment vs. contradiction for each label independently
func getMultiClassScores(logits []*mat.Dense, entailmentID, contradictionID int) []mat.Float {
	scores := make([]mat.Float, len(logits))
	for i, v := range logits {
		prob := floatutils.SoftMax([]mat.Float{v.AtVec(entailmentID), v.AtVec(contradictionID)})
		scores[i] = prob[0]
	}
	return scores
}

// getScores softmax the "entailment" over all candidate labels
func getScores(logits []*mat.Dense, entailmentID int) []mat.Float {
	scores := make([]mat.Float, len(logits))
	for i, l := range logits {
		scores[i] = l.AtVec(entailmentID)
	}
	return floatutils.SoftMax(scores)
}

func (s *ServerForSequenceClassification) getEntailmentAndContradictionIDs() (
	entailmentID, contradictionID int, err error,
) {
	entailmentID, ok := s.model.BART.Config.Label2ID["entailment"]
	if !ok {
		return -1, -1, errors.New("bartserver: `entailment` label not found")
	}
	contradictionID, ok = s.model.BART.Config.Label2ID["contradiction"]
	if !ok {
		return -1, -1, errors.New("bartserver: `contradiction` label not found")
	}
	return
}

func (s *ServerForSequenceClassification) newWorkers(workersSize int) []*worker {
	workers := make([]*worker, workersSize)
	for i := range workers {
		workers[i] = &worker{
			tokenizer: s.tokenizer,
			model:     s.model,
		}
	}
	return workers
}

type worker struct {
	tokenizer *bpetokenizer.BPETokenizer
	model     *barthead.SequenceClassification
}

func (w *worker) process(input premiseHypothesisPair) *mat.Dense {
	g := ag.NewGraph(ag.ConcurrentComputations(runtime.NumCPU()), ag.IncrementalForward(false))
	defer g.Clear()
	proc := nn.Reify(nn.Context{Graph: g, Mode: nn.Inference}, w.model).(*barthead.SequenceClassification)
	inputIds := getInputIDs(w.tokenizer, input.premise, input.hypothesis)
	logits := proc.Classify(inputIds)
	g.Forward()
	return g.GetCopiedValue(logits).(*mat.Dense)
}
