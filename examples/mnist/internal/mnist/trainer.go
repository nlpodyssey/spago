// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mnist

import (
	"fmt"
	"github.com/gosuri/uiprogress"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/losses"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
	"github.com/nlpodyssey/spago/pkg/utils"
	"github.com/nlpodyssey/spago/pkg/utils/data"
	"sync"
)

type Trainer struct {
	model       nn.Model
	optimizer   *gd.GradientDescent
	epochs      int
	batchSize   int
	concurrent  bool
	trainSet    Dataset
	testSet     Dataset
	modelPath   string
	curLoss     float64
	curEpoch    int
	indices     []int            // the indices of the train-set; used to shuffle the examples each epoch
	rndShuffler *rand.LockedRand // the random generator used to shuffle the examples
	rndSeed     uint64
}

func NewTrainer(
	model nn.Model,
	optimizer *gd.GradientDescent,
	epochs int,
	batchSize int,
	concurrent bool,
	trainSet Dataset,
	testSet Dataset,
	modelPath string,
	seed uint64,
) *Trainer {
	return &Trainer{
		model:       model,
		optimizer:   optimizer,
		epochs:      epochs,
		batchSize:   batchSize,
		concurrent:  concurrent,
		trainSet:    trainSet,
		testSet:     testSet,
		modelPath:   modelPath,
		curLoss:     0,
		curEpoch:    0,
		indices:     utils.MakeIndices(trainSet.Count()),
		rndShuffler: rand.NewLockedRand(seed),
		rndSeed:     seed,
	}
}

func (t *Trainer) newTrainBar(progress *uiprogress.Progress) *uiprogress.Bar {
	bar := progress.AddBar(t.trainSet.Count())
	bar.AppendCompleted().PrependElapsed()
	bar.PrependFunc(func(b *uiprogress.Bar) string {
		return fmt.Sprintf("Epoch: %d Loss: %.6f", t.curEpoch, t.curLoss)
	})
	return bar
}

func (t *Trainer) Enjoy() {
	for epoch := 0; epoch < t.epochs; epoch++ {
		t.curEpoch = epoch
		t.optimizer.IncEpoch()

		fmt.Println("Training epoch...")
		t.trainEpoch()

		fmt.Println("Start evaluation...")
		precision := NewEvaluator(t.model).Evaluate(t.testSet).Precision()
		fmt.Printf("Accuracy: %.2f\n", 100*precision)

		// model serialization
		err := utils.SerializeToFile(t.modelPath, nn.NewParamsSerializer(t.model))
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
	rand.ShuffleInPlace(t.indices, t.rndShuffler)
	batchId := 0
	data.ForEachBatch(len(t.indices), t.batchSize, func(start, end int) {
		batchId++
		t.trainBatch(batchId, t.indices[start:end], func() { bar.Incr() })
		t.optimizer.Optimize()
	})
}

func (t *Trainer) trainBatch(batchId int, indices []int, onExample func()) {
	if t.concurrent {
		t.trainBatchConcurrent(batchId, indices, onExample)
	} else {
		t.trainBatchSerial(batchId, indices, onExample)
	}
}

func (t *Trainer) trainBatchConcurrent(batchId int, indices []int, onExample func()) {
	t.optimizer.IncBatch()
	var wg sync.WaitGroup
	wg.Add(len(indices))
	for _, i := range indices {
		t.optimizer.IncExample()
		go func(example *Example) {
			defer wg.Done()
			defer onExample()
			t.curLoss = t.learn(batchId, example)
		}(t.trainSet.GetExample(i))
	}
	wg.Wait()
}

func (t *Trainer) trainBatchSerial(batchId int, indices []int, onExample func()) {
	t.optimizer.IncBatch()
	for _, i := range indices {
		t.optimizer.IncExample()
		t.curLoss = t.learn(batchId, t.trainSet.GetExample(i))
		onExample()
	}
}

// learn performs the backward respect to the cross-entropy loss, returned as scalar value
func (t *Trainer) learn(_ int, example *Example) float64 {
	g := ag.NewGraph(ag.Rand(rand.NewLockedRand(t.rndSeed)))
	defer g.Clear()
	x := g.NewVariable(example.Features, false)
	y := t.model.NewProc(g).Forward(x)[0]
	loss := g.Div(losses.CrossEntropy(g, y, example.Label), g.NewScalar(float64(t.batchSize)))
	g.Backward(loss)
	return loss.ScalarValue()
}
