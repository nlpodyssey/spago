// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package charlm

import (
	"fmt"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/losses"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd/gdmbuilder"
	"github.com/nlpodyssey/spago/pkg/nlp/corpora"
	"github.com/nlpodyssey/spago/pkg/utils"
	"runtime"
)

// TrainingConfig provides configuration settings for a Character-level Language Trainer.
// TODO: add dropout
type TrainingConfig[T mat.DType] struct {
	Seed                  uint64
	BatchSize             int
	BackStep              int
	GradientClipping      T
	SerializationInterval int
	UpdateMethod          gd.MethodConfig
	ModelPath             string
}

// Trainer implements the training process for a Character-level Language Model.
type Trainer[T mat.DType] struct {
	TrainingConfig[T]
	randGen       *rand.LockedRand[T]
	corpus        corpora.TextCorpusIterator
	model         *Model[T]
	optimizer     *gd.GradientDescent[T]
	bestLoss      T
	lastBatchLoss T
	curPerplexity T
}

// NewTrainer returns a new Trainer.
func NewTrainer[T mat.DType](config TrainingConfig[T], corpus corpora.TextCorpusIterator, model *Model[T]) *Trainer[T] {
	return &Trainer[T]{
		TrainingConfig: config,
		randGen:        rand.NewLockedRand[T](config.Seed),
		corpus:         corpus,
		model:          model,
		optimizer: gd.NewOptimizer[T](
			gdmbuilder.NewMethod[T](config.UpdateMethod),
			nn.NewDefaultParamsIterator[T](model),
			gd.ClipGradByNorm[T](config.GradientClipping, 2.0)),
	}
}

// Train executes the training process.
func (t *Trainer[T]) Train() {
	t.corpus.ForEachLine(func(i int, line string) {
		t.trainPassage(i, line)
		// TODO: save the model only if it is better against a validation criterion (yet to be defined)
		if i > 0 && i%t.SerializationInterval == 0 {
			fmt.Println("=== MODEL SERIALIZATION")
			err := utils.SerializeToFile(t.ModelPath, t.model)
			if err != nil {
				panic("charlm: error during model serialization.")
			}
		}
	})
}

func (t *Trainer[T]) trainPassage(index int, text string) {
	// This is a particular case where computing the forward after the graph definition can be more efficient.
	g := ag.NewGraph[T](
		ag.Rand(t.randGen),
		ag.IncrementalForward[T](false),
		ag.ConcurrentComputations[T](runtime.NumCPU()),
	)
	defer g.Clear()
	proc := nn.ReifyForTraining(t.model, g)

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
		t.curPerplexity = mat.Exp(loss)
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
func (t *Trainer[T]) trainBatch(proc *Model[T], batch []string) T {
	g := proc.Graph()
	g.ZeroGrad()
	prevTimeStep := g.TimeStep()
	predicted := proc.Forward(batch).([]ag.Node[T])
	targets := targetsIds(batch, t.model.Vocabulary, t.model.UnknownToken)
	loss := losses.CrossEntropySeq(g, predicted[:len(targets)], targets, true)
	g.Forward(ag.Range[T](prevTimeStep+1, -1))
	g.Backward(loss, ag.Truncate[T](t.BackStep))
	return loss.ScalarValue()
}
