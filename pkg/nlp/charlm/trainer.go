// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package charlm

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/losses"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd/gdmbuilder"
	"github.com/nlpodyssey/spago/pkg/nlp/corpora"
	"github.com/nlpodyssey/spago/pkg/utils"
	"math"
)

// TODO: add dropout
type TrainingConfig struct {
	Seed                  uint64
	BatchSize             int
	BackStep              int
	GradientClipping      float64
	SerializationInterval int
	UpdateMethod          gd.MethodConfig
	ModelPath             string
}

type Trainer struct {
	TrainingConfig
	randGen       *rand.LockedRand
	corpus        corpora.TextCorpusIterator
	model         *Model
	optimizer     *gd.GradientDescent
	bestLoss      float64
	lastBatchLoss float64
	curPerplexity float64
}

func NewTrainer(config TrainingConfig, corpus corpora.TextCorpusIterator, model *Model) *Trainer {
	return &Trainer{
		TrainingConfig: config,
		randGen:        rand.NewLockedRand(config.Seed),
		corpus:         corpus,
		model:          model,
		optimizer: gd.NewOptimizer(
			gdmbuilder.NewMethod(config.UpdateMethod),
			nn.NewDefaultParamsIterator(model),
			gd.ClipGradByNorm(config.GradientClipping, 2.0)),
	}
}

func (t *Trainer) Train() {
	t.corpus.ForEachLine(func(i int, line string) {
		t.trainPassage(i, line)
		// TODO: save the model only if it is better against a validation criterion (yet to be defined)
		if i > 0 && i%t.SerializationInterval == 0 {
			fmt.Println("=== MODEL SERIALIZATION")
			err := utils.SerializeToFile(t.ModelPath, nn.NewParamsSerializer(t.model))
			if err != nil {
				panic("charlm: error during model serialization.")
			}
		}
	})
}

func (t *Trainer) trainPassage(index int, text string) {
	// This is a particular case where computing the forward after the graph definition can be more efficient.
	g := ag.NewGraph(
		ag.Rand(t.randGen),
		ag.IncrementalForward(false),
		ag.ConcurrentComputations(true),
	)
	defer g.Clear()
	proc := t.model.NewProc(nn.Context{Graph: g, Mode: nn.Training}).(*Processor)

	// Split the text into runes and append the sequence separator
	sequence := utils.SplitByRune(text)
	sequence = append(sequence, t.model.SequenceSeparator)

	length := len(sequence)
	cnt := 0
	for i := 0; i < length; i += t.BatchSize {
		j := i + t.BatchSize
		if j > length {
			j = length
		}
		batch := sequence[i:j]
		if len(batch) == 1 {
			break // there is no subsequent character to predict, nothing more to learn.
		}
		cnt += len(batch)
		t.optimizer.IncBatch()
		t.optimizer.IncExample()
		loss := t.trainBatch(proc, batch)
		t.optimizer.Optimize()
		t.lastBatchLoss = loss
		t.curPerplexity = math.Exp(loss)
	}
	if g.TimeStep() != cnt {
		panic(fmt.Sprintf("charlm: time-step `%d` different than processed items `%d`. Something goes wrong.",
			g.TimeStep(), cnt))
	}

	// TODO: improve print
	fmt.Println(text)
	fmt.Printf("Cnt: %d Sentene length: %d Loss: %.6f Perplexity: %.6f\n\n",
		index, length, t.lastBatchLoss, t.curPerplexity)
}

// trainBatch performs both the forward step and the truncated back-propagation on a given batch.
// Note that the processor remains the same for all batches of the same sequence,
// so the previous recurrent states are retained for the next prediction.
func (t *Trainer) trainBatch(proc *Processor, batch []string) float64 {
	g := proc.GetGraph()
	g.ZeroGrad()
	prevTimeStep := g.TimeStep()
	predicted := proc.Predict(batch...)
	targets := targetsIds(batch, t.model.Vocabulary, t.model.UnknownToken)
	loss := losses.CrossEntropySeq(g, predicted[:len(targets)], targets, true)
	g.Forward(ag.Range(prevTimeStep+1, -1))
	g.Backward(loss, ag.Truncate(t.BackStep))
	return loss.ScalarValue()
}
