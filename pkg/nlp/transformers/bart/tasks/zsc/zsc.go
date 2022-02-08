// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zsc

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/matutils"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/nlp/tokenizers/bpetokenizer"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/head/sequenceclassification"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/loader"
	"github.com/nlpodyssey/spago/pkg/nlp/transformers/bart/tasks"
	"github.com/nlpodyssey/spago/pkg/utils/workerpool"
	"runtime"
	"sort"
	"strings"
	"sync"
)

var _ nn.Model = &BartForZeroShotClassification{}

// BartForZeroShotClassification combines a sequence classification BART
// model with a BPE tokenizer to perform Zero-Shot Text Classification.
type BartForZeroShotClassification struct {
	*sequenceclassification.Model
	Tokenizer *bpetokenizer.BPETokenizer
}

// LoadModel loads a BartForZeroShotClassification from file.
func LoadModel(modelPath string) (*BartForZeroShotClassification, error) {
	model, err := loader.Load(modelPath)
	if err != nil {
		return nil, err
	}
	tokenizer, err := bpetokenizer.NewFromModelFolder(modelPath)
	if err != nil {
		return nil, err
	}
	return &BartForZeroShotClassification{
		Model:     model.(*sequenceclassification.Model),
		Tokenizer: tokenizer,
	}, nil
}

type premiseHypothesisPair struct {
	index      int
	premise    string
	hypothesis string
}

const (
	defaultHypothesisTemplate   = "This text is about {}."
	defaultStartSequenceTokenID = 0
	defaultEndSequenceTokenID   = 2
)

// Classify performs a text classification using the zero-shot technique.
func (t *BartForZeroShotClassification) Classify(
	text string,
	hypothesisTemplate string,
	candidateLabels []string,
	multiClass bool,
) (*tasks.ClassifyResponse, error) {
	if hypothesisTemplate == "" {
		hypothesisTemplate = defaultHypothesisTemplate
	}

	entailmentID, contradictionID, err := t.getEntailmentAndContradictionIDs()
	if err != nil {
		return nil, err
	}

	numOfCandidateLabels := len(candidateLabels)
	logits := make([]mat.Matrix[mat.Float], numOfCandidateLabels)

	numWorkers := runtime.NumCPU() / 2 // leave some space for other concurrent computations
	wp := workerpool.New(numWorkers)
	workers := t.newWorkers(numWorkers)
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

	best := matutils.ArgMax(scores)
	class := candidateLabels[best]

	distribution := make([]tasks.ClassConfidencePair, len(scores))
	for i := 0; i < len(scores); i++ {
		distribution[i] = tasks.ClassConfidencePair{
			Class:      candidateLabels[i],
			Confidence: scores[i],
		}
	}

	sort.Slice(distribution, func(i, j int) bool {
		return distribution[i].Confidence > distribution[j].Confidence
	})

	return &tasks.ClassifyResponse{
		Class:        class,
		Confidence:   scores[best],
		Distribution: distribution,
	}, nil
}

// getMultiClassScores softmax over the entailment vs. contradiction for each label independently
func getMultiClassScores(logits []mat.Matrix[mat.Float], entailmentID, contradictionID int) []mat.Float {
	scores := make([]mat.Float, len(logits))
	for i, v := range logits {
		prob := matutils.SoftMax([]mat.Float{v.AtVec(entailmentID), v.AtVec(contradictionID)})
		scores[i] = prob[0]
	}
	return scores
}

// getScores softmax the "entailment" over all candidate labels
func getScores(logits []mat.Matrix[mat.Float], entailmentID int) []mat.Float {
	scores := make([]mat.Float, len(logits))
	for i, l := range logits {
		scores[i] = l.AtVec(entailmentID)
	}
	return matutils.SoftMax(scores)
}

func (t *BartForZeroShotClassification) getEntailmentAndContradictionIDs() (
	entailmentID, contradictionID int, err error,
) {
	labels2id := t.Model.BART.Config.Label2ID
	var ok bool
	entailmentID, ok = labels2id["entailment"]
	if !ok {
		return -1, -1, fmt.Errorf("server: `entailment` label not found")
	}
	contradictionID, ok = labels2id["contradiction"]
	if !ok {
		return -1, -1, fmt.Errorf("server: `contradiction` label not found")
	}
	return
}

func (t *BartForZeroShotClassification) newWorkers(workersSize int) []*worker {
	workers := make([]*worker, workersSize)
	for i := range workers {
		workers[i] = &worker{
			tokenizer: t.Tokenizer,
			model:     t.Model,
		}
	}
	return workers
}

type worker struct {
	tokenizer *bpetokenizer.BPETokenizer
	model     *sequenceclassification.Model
}

func (w *worker) process(input premiseHypothesisPair) mat.Matrix[mat.Float] {
	g := ag.NewGraph(ag.ConcurrentComputations(runtime.NumCPU()), ag.IncrementalForward(false))
	defer g.Clear()
	proc := nn.ReifyForInference(w.model, g)
	inputIds := getInputIDs(w.tokenizer, input.premise, input.hypothesis)
	logits := proc.Classify(inputIds)
	g.Forward()
	return g.GetCopiedValue(logits)
}

func getInputIDs(tokenizer *bpetokenizer.BPETokenizer, text, text2 string) []int {
	encoded, _ := tokenizer.Encode(text) // TODO: error handling
	inputIds := append(append([]int{defaultStartSequenceTokenID}, encoded.IDs...), defaultEndSequenceTokenID)
	if text2 != "" {
		encoded2, _ := tokenizer.Encode(text2) // TODO: error handling
		inputIds2 := append(append([]int{defaultEndSequenceTokenID}, encoded2.IDs...), defaultEndSequenceTokenID)
		inputIds = append(inputIds, inputIds2...)
	}
	return inputIds
}
