// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package de

import (
	"golang.org/x/exp/rand"
	"saientist.dev/spago/pkg/mat"
)

// Differential Evolution implements a simple and efficient heuristic for global optimization over continuous spaces.
// Reference: https://link.springer.com/article/10.1023/A:1008202821328 (Storn & Price, 1997)
type DifferentialEvolution struct {
	// The initial configuration
	Config
	// The population of the current generation
	population *Population
	// The mutation strategy
	mutation Mutator
	// The crossover strategy
	crossover Crossoverer
	// The fitness function to minimize
	fitnessFunc func(solution *mat.Dense, batch int) float64
	// The validation function to maximize
	validate func(solution *mat.Dense) float64
	// Method to call after finding a new best solution
	onNewBest func(solution *ScoredVector)
	// The current best solution (can be nil)
	bestSolution *ScoredVector
	// Optimization state
	state *State
}

type State struct {
	// The current generation
	CurGeneration int
	// The current batch
	CurBatch int
	// The current optimization step
	CurOptimizationStep int
	// Count the times the best solution remains unchanged
	countBestScoreUnchanged int
}

type Config struct {
	// The number of member of the populations
	PopulationSize int
	// The size of the dense vector
	VectorSize int
	// The maximum number of generations over which the entire population is evolved
	MaxGenerations int
	// The number of batches used to calculate the scores of the TargetVector and Trial vectors
	BatchSize int
	// The number of optimization steps to do for each generation
	OptimizationSteps int
	// Differential weight (default 0.5)
	MutationFactor float64
	// Crossover probability (default 0.9)
	CrossoverRate float64
	// The (positive) bound
	Bound float64
	// Whether to alter the mutation factor and the crossover rate on the Trial evaluation
	Adaptive bool
	// Reset the population if the best solution remains unchanged after for this long
	ResetAfter int
	// The random seed.
	Seed uint64
}

type ScoredVector struct {
	Vector *mat.Dense
	Score  float64
}

// NewOptimizer returns a new DifferentialEvolution ready to optimize your problem.
func NewOptimizer(
	config Config,
	mutation Mutator,
	crossover Crossoverer,
	score func(solution *mat.Dense, batch int) float64,
	validate func(solution *mat.Dense) float64,
	onNewBest func(solution *ScoredVector),
) *DifferentialEvolution {
	return &DifferentialEvolution{
		Config: config,
		population: NewRandomPopulation(
			config.PopulationSize,
			config.VectorSize,
			config.Bound,
			rand.NewSource(config.Seed),
			MemberHyperParams{
				MutationFactor: config.MutationFactor,
				CrossoverRate:  config.CrossoverRate,
			}),
		mutation:     mutation,
		crossover:    crossover,
		fitnessFunc:  score,
		validate:     validate,
		onNewBest:    onNewBest,
		bestSolution: nil,
		state: &State{
			CurBatch:                0,
			CurGeneration:           0,
			CurOptimizationStep:     0,
			countBestScoreUnchanged: 0,
		},
	}
}

// Optimize performs the Differential Evolution optimization process.
func (o *DifferentialEvolution) Optimize() {
	panic("de: method not implemented") // TODO
}
