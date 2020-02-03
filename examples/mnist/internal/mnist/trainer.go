// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mnist

import (
	"fmt"
	"github.com/gosuri/uiprogress"
	"golang.org/x/exp/rand"
	"runtime/debug"
	"saientist.dev/spago/pkg/mat"
	"saientist.dev/spago/pkg/mat/rnd"
	"saientist.dev/spago/pkg/ml/ag"
	"saientist.dev/spago/pkg/ml/losses"
	"saientist.dev/spago/pkg/ml/nn"
	"saientist.dev/spago/pkg/ml/optimizers/gd"
	"saientist.dev/spago/pkg/utils"
	"saientist.dev/spago/third_party/GoMNIST"
	"sync"
)

type Trainer struct {
	model      nn.Model
	optimizer  *gd.GradientDescent
	epochs     int
	batchSize  int
	concurrent bool
	trainSet   Dataset
	testSet    Dataset
	modelPath  string
	curLoss    float64
	curEpoch   int
	indices    []int       // the indices of the train-set; used to shuffle the examples each epoch
	rndSrc     rand.Source // the random source used to shuffle the examples
}

func NewTrainer(model nn.Model, optimizer *gd.GradientDescent, epochs, batchSize int, concurrent bool, trainSet, testSet Dataset, modelPath string, rndSrc rand.Source) *Trainer {
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
		indices:    utils.MakeIndices(trainSet.Count()),
		rndSrc:     rndSrc,
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

		debug.FreeOSMemory() // a lot of things have happened, better to tap the gc
	}
}

func (t *Trainer) trainEpoch() {
	uip := uiprogress.New()
	bar := t.newTrainBar(uip)
	uip.Start() // start bar rendering
	defer uip.Stop()

	rnd.ShuffleInPlace(t.indices, t.rndSrc)
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
		go func(image *mat.Dense, label GoMNIST.Label) {
			defer wg.Done()
			defer onExample()
			t.curLoss = t.learn(image, int(label))
		}(t.trainSet.GetNormalized(i))
	}
	wg.Wait()
}

func (t *Trainer) trainBatchSerial(indices []int, onExample func()) {
	t.optimizer.IncBatch()
	for _, i := range indices {
		t.optimizer.IncExample()
		image, label := t.trainSet.GetNormalized(i)
		t.curLoss = t.learn(image, int(label))
		onExample()
	}
}

// learn performs the backward respect to the cross-entropy loss, returned as scalar value
func (t *Trainer) learn(image *mat.Dense, label int) float64 {
	g := ag.NewGraph()
	x := g.NewVariable(image, false)
	y := t.model.NewProc(g).Forward(x)[0]
	loss := g.Div(losses.CrossEntropy(g, y, label), g.NewScalar(float64(t.batchSize)))
	g.Backward(loss)
	return loss.ScalarValue()
}
