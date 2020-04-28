// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"github.com/nlpodyssey/spago/examples/mnist/internal/mnist"
	"github.com/nlpodyssey/spago/examples/mnist/third_party/GoMNIST"
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/mat/f64utils"
	"github.com/nlpodyssey/spago/pkg/mat/rand"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/losses"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"github.com/nlpodyssey/spago/pkg/ml/nn/perceptron"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/de"
	"github.com/nlpodyssey/spago/pkg/ml/stats"
	"github.com/nlpodyssey/spago/pkg/utils"
	"github.com/nlpodyssey/spago/pkg/utils/data"
	"log"
	"net/http"
	_ "net/http/pprof"
	"os"
	"sync"
)

func main() {
	// go tool pprof http://localhost:6060/debug/pprof/profile
	go func() { log.Println(http.ListenAndServe("localhost:6060", nil)) }()

	modelPath := os.Args[1]

	var datasetPath string
	if len(os.Args) > 2 {
		datasetPath = os.Args[2]
	} else {
		// assuming default path
		datasetPath = "examples/mnist/third_party/GoMNIST/data"
	}

	// read dataset
	trainSet, testSet, err := GoMNIST.Load(datasetPath)
	if err != nil {
		panic("Error reading MNIST data.")
	}

	trainExamples := mnist.GetAllExamples(trainSet)
	validationExample := mnist.GetAllExamples(testSet) // TODO: use the 10% of the training set!
	trainBatches := data.GenerateBatches(len(trainExamples), 20, func(i int) int { return trainExamples[i].Label })

	// new template model initialized with zeros
	modelFactory := func() nn.Model {
		return perceptron.New(
			784,           // features
			10,            // output
			ag.OpIdentity, // output activation
		)
	}

	fitnessCallback := (&fitnessFunc{
		modelFactory: modelFactory,
		examples:     trainExamples,
		batches:      trainBatches,
	}).callback

	validateCallback := (&validator{
		modelFactory: modelFactory,
		examples:     validationExample,
	}).callback

	onNewBestCallback := (&onNewBest{
		modelFactory: modelFactory,
		modelPath:    modelPath,
		examples:     validationExample,
	}).callback

	//mutationStrategy := de.NewRandomMutation(6.0) // this is the base mutation strategy
	mutationStrategy := de.NewDeglMutation(0.1, 6.0) // this is a more advanced mutation strategy
	crossoverStrategy := de.NewBinomialCrossover(rand.NewLockedRand(uint64(42)))

	optimizer := de.NewOptimizer(
		de.Config{
			PopulationSize:    400,
			VectorSize:        nn.DumpParamsVector(modelFactory()).Size(),
			MaxGenerations:    50,
			BatchSize:         len(trainBatches),
			OptimizationSteps: 200,
			MutationFactor:    0.5,
			CrossoverRate:     0.9,
			WeightFactor:      0.5,
			Bound:             6.0,
			Adaptive:          true,
			ResetAfter:        10,
			Seed:              42,
		},
		mutationStrategy,
		crossoverStrategy,
		fitnessCallback,
		validateCallback,
		onNewBestCallback,
	)

	fmt.Println("Let the optimization begin!")
	optimizer.Optimize() // Enjoy :)
}

type fitnessFunc struct {
	modelFactory func() nn.Model
	examples     []*mnist.Example
	batches      [][]int
}

func (r *fitnessFunc) callback(solution *mat.Dense, batchIndex int) float64 {
	batch := r.batches[batchIndex]
	batchSize := len(batch)
	model := r.modelFactory()
	nn.LoadParamsVector(model, solution)
	batchLosses := make([]float64, batchSize)
	var wg sync.WaitGroup
	wg.Add(batchSize)
	for i, exampleIndex := range batch {
		go func(i int, example *mnist.Example) {
			defer wg.Done()
			g := ag.NewGraph()
			x := g.NewVariable(example.Features, false)
			y := model.NewProc(g).Forward(x)[0]
			batchLosses[i] = losses.CrossEntropy(g, y, example.Label).ScalarValue()
		}(i, r.examples[exampleIndex])
	}
	wg.Wait()
	return mat.NewVecDense(batchLosses).Sum() / float64(batchSize)
}

type validator struct {
	modelFactory func() nn.Model
	examples     []*mnist.Example
}

func (r *validator) callback(solution *mat.Dense) float64 {
	model := r.modelFactory()
	nn.LoadParamsVector(model, solution)
	counter := stats.NewMetricCounter()
	for _, example := range r.examples {
		g := ag.NewGraph()
		x := g.NewVariable(example.Features, false)
		y := model.NewProc(g).Forward(x)[0]
		argMax := f64utils.ArgMax(y.Value().Data())
		if argMax == example.Label {
			counter.IncTruePos()
		} else {
			counter.IncFalsePos()
		}
	}
	return counter.Precision()
}

type onNewBest struct {
	modelFactory func() nn.Model
	modelPath    string
	examples     []*mnist.Example
}

func (r *onNewBest) callback(solution *de.ScoredVector) {
	model := r.modelFactory()
	nn.LoadParamsVector(model, solution.Vector)
	counter := stats.NewMetricCounter()
	for _, example := range r.examples {
		g := ag.NewGraph()
		x := g.NewVariable(example.Features, false)
		y := model.NewProc(g).Forward(x)[0]
		argMax := f64utils.ArgMax(y.Value().Data())
		if argMax == example.Label {
			counter.IncTruePos()
		} else {
			counter.IncFalsePos()
		}
	}
	fmt.Printf("Accuracy: %.2f\n", 100*counter.Precision())
	fmt.Printf("Saving model to \"%s\"... ", r.modelPath)
	err := utils.SerializeToFile(r.modelPath, model)
	if err != nil {
		panic("mnist: error during model serialization.")
	}
	fmt.Println("OK")
}
