// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package internal

import (
	"fmt"
	"github.com/gosuri/uiprogress"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/losses"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
	"github.com/nlpodyssey/spago/pkg/utils"
	"sync"
)

type Trainer struct {
	model      nn.Model
	optimizer  *gd.GradientDescent
	epochs     int
	batchSize  int
	concurrent bool
	trainSet   []Sequence
	testSet    []Sequence
	modelPath  string
	curLoss    float64
	curEpoch   int
	indices    []int            // the indices of the train-set; used to shuffle the examples each epoch
	rndGen     *rand.LockedRand // the random source used to shuffle the examples
}

func NewTrainer(
	model nn.Model,
	optimizer *gd.GradientDescent,
	epochs, batchSize int,
	concurrent bool,
	trainSet, testSet []Sequence,
	modelPath string,
	rndGen *rand.LockedRand,
) *Trainer {
	return &Trainer{
		model:      model,
		optimizer:  optimizer,
		epochs:     epochs,
		batchSize:  batchSize,
		concurrent: concurrent,
		trainSet:   trainSet,
		testSet:    testSet,
		modelPath:  modelPath,
		curLoss:    0,
		curEpoch:   0,
		indices:    utils.MakeIndices(len(trainSet)),
		rndGen:     rndGen,
	}
}

func (t *Trainer) newTrainBar(progress *uiprogress.Progress) *uiprogress.Bar {
	bar := progress.AddBar(len(t.trainSet))
	bar.AppendCompleted().PrependElapsed()
	bar.PrependFunc(func(b *uiprogress.Bar) string {
		return fmt.Sprintf("Epoch: %d Loss: %.6f", t.curEpoch, t.curLoss)
	})
	return bar
}

func (t *Trainer) Enjoy() {
	nn.TrackParams(t.model, t.optimizer)
	for epoch := 0; epoch < t.epochs; epoch++ {
		t.curEpoch = epoch
		t.optimizer.IncEpoch()
		fmt.Println("Training epoch...")
		t.trainEpoch()
		fmt.Println("Start evaluation...")
		precision := NewEvaluator(t.model).Evaluate(t.testSet).Precision()
		fmt.Printf("Accuracy: %.2f\n", 100*precision)
		// model serialization
		err := utils.SerializeToFile(t.modelPath, t.model)
		if err != nil {
			panic("mnist: error during model serialization.")
		}
	}
}

func (t *Trainer) trainEpoch() {
	uip := uiprogress.New()
	bar := t.newTrainBar(uip)
	uip.Start() // start bar rendering
	defer uip.Stop()

	rand.ShuffleInPlace(t.indices, t.rndGen)
	for start := 0; start < len(t.indices); start += t.batchSize {
		end := utils.MinInt(start+t.batchSize, len(t.indices)-1)
		t.trainBatch(t.indices[start:end], func() { bar.Incr() })
		t.optimizer.Optimize()
	}
}

func (t *Trainer) trainBatch(indices []int, onExample func()) {
	if t.concurrent {
		t.trainBatchConcurrent(indices, onExample)
	} else {
		t.trainBatchSerial(indices, onExample)
	}
}

func (t *Trainer) trainBatchConcurrent(indices []int, onExample func()) {
	t.optimizer.IncBatch()
	var wg sync.WaitGroup
	wg.Add(len(indices))
	for _, i := range indices {
		t.optimizer.IncExample()
		go func(example Sequence) {
			defer wg.Done()
			defer onExample()
			t.curLoss = t.learn(example)
		}(t.trainSet[i])
	}
	wg.Wait()
}

func (t *Trainer) trainBatchSerial(indices []int, onExample func()) {
	t.optimizer.IncBatch()
	for _, i := range indices {
		t.optimizer.IncExample()
		t.curLoss = t.learn(t.trainSet[i])
		onExample()
	}
}

// learn performs the backward respect to the cross-entropy loss, returned as scalar value
func (t *Trainer) learn(example Sequence) float64 {
	g := ag.NewGraph()
	defer g.Clear()
	xs, ts := extract(g, example)
	ys := t.model.NewProc(g).Forward(xs...)
	loss := g.Div(losses.CrossEntropySeq(g, ys, ts, false), g.NewScalar(float64(t.batchSize)))
	g.Backward(loss)
	return loss.ScalarValue()
}

func extract(g *ag.Graph, example Sequence) (xs []ag.Node, targets []int) {
	xs = make([]ag.Node, len(example))
	targets = make([]int, len(example))
	for i, x := range example {
		xs[i] = g.NewScalar(x.Input)
		targets[i] = x.Target
	}
	return
}
