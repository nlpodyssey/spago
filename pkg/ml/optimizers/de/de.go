// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package de

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/mat32/rand"
)

// DifferentialEvolution implements a simple and efficient heuristic for global optimization over continuous spaces.
// Reference: https://link.springer.com/article/10.1023/A:1008202821328 (Storn & Price, 1997)
type DifferentialEvolution struct {
	// The initial configuration
	Config
	// The population of the current generation
	population *Population
	// The mutation strategy
	mutation Mutator
	// The crossover strategy
	crossover Crossover
	// The fitness function to minimize
	fitnessFunc func(solution mat.Matrix, batch int) mat.Float
	// The validation function to maximize
	validate func(solution mat.Matrix) mat.Float
	// Method to call after finding a new best solution
	onNewBest func(solution *ScoredVector)
	// The current best solution (can be nil)
	bestSolution *ScoredVector
	// Optimization state
	state *State
}

// State represents a status of the differential evolution process.
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

// Config provides configuration settings for a DifferentialEvolution optimizer.
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
	MutationFactor mat.Float
	// Crossover probability (default 0.9)
	CrossoverRate mat.Float
	// Weight factor used by DEGL mutation strategy (default 0.5)
	WeightFactor mat.Float
	// The (positive) bound
	Bound mat.Float
	// Whether to alter the mutation factor and the crossover rate on the Trial evaluation
	Adaptive bool
	// Reset the population if the best solution remains unchanged after for this long
	ResetAfter int
	// The random seed.
	Seed uint64
}

// ScoredVector is a pair which associates a Score to a Vector corresponding to a specific solution.
type ScoredVector struct {
	Vector mat.Matrix
	Score  mat.Float
}

// NewOptimizer returns a new DifferentialEvolution ready to optimize your problem.
func NewOptimizer(
	config Config,
	mutation Mutator,
	crossover Crossover,
	score func(solution mat.Matrix, batch int) mat.Float,
	validate func(solution mat.Matrix) mat.Float,
	onNewBest func(solution *ScoredVector),
) *DifferentialEvolution {
	return &DifferentialEvolution{
		Config: config,
		population: NewRandomPopulation(
			config.PopulationSize,
			config.VectorSize,
			config.Bound,
			rand.NewLockedRand(config.Seed),
			MemberHyperParams{
				MutationFactor: config.MutationFactor,
				CrossoverRate:  config.CrossoverRate,
				WeightFactor:   config.WeightFactor,
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
	for g := 0; g < o.MaxGenerations; g++ {
		o.state.CurGeneration = g
		for batch := 0; batch < o.BatchSize; batch++ {
			o.state.CurBatch = batch
			o.optimizeBatch()
		}
	}
}

// optimizeBatch optimize the current generation against the current batch.
func (o *DifferentialEvolution) optimizeBatch() {
	o.evaluateTargets()
	o.optimizeGeneration()
	o.validateTargets()
	o.checkForBetterSolution()
	if o.ResetAfter > o.state.countBestScoreUnchanged {
		o.resetPopulation()
	}
}

// optimizeGenerations performs n optimization steps on the current generation, as many times as defined in the configuration.
func (o *DifferentialEvolution) optimizeGeneration() {
	for i := 0; i < o.OptimizationSteps; i++ {
		o.state.CurOptimizationStep = i
		o.optimizationStep()
	}
}

// optimizationStep performs the mutation, the crossover and the trial evaluation.
func (o *DifferentialEvolution) optimizationStep() {
	o.mutation.Mutate(o.population)
	o.crossover.Crossover(o.population)
	o.evaluateTrials()
}

// evaluateTargets evaluate the fitness of the target vectors against the current batch for each member of the population.
func (o *DifferentialEvolution) evaluateTargets() {
	for _, member := range o.population.Members {
		member.TargetScore = o.fitnessFunc(member.TargetVector, o.state.CurBatch)
	}
}

// evaluateTrials evaluate the fitness of the donor vectors against the current batch for each member of the population.
// If the fitness is better than the current one, assign the value of the donor vector to the target vector.
func (o *DifferentialEvolution) evaluateTrials() {
	for _, member := range o.population.Members {
		member.TrialScore = o.fitnessFunc(member.DonorVector, o.state.CurBatch)
		if member.TrialScore < member.TargetScore {
			member.TargetScore = member.TrialScore
			member.TargetVector = member.DonorVector.Clone()
			if o.Adaptive {
				member.MutateHyperParams(0.1, 0.9) // TODO: get arguments from the config
			}
		}
	}
}

// validateTargets test the entire population against the validation dataset.
func (o *DifferentialEvolution) validateTargets() {
	for _, member := range o.population.Members {
		member.ValidationScore = o.validate(member.TargetVector)
	}
}

// checkForBetterSolution compares the overall best solution with all current solutions, updating it if a new best is found.
func (o *DifferentialEvolution) checkForBetterSolution() {
	bestIndex := 0
	bestValidationScore := mat.Inf(-1)
	for i, member := range o.population.Members {
		if member.ValidationScore > bestValidationScore {
			bestValidationScore = member.ValidationScore
			bestIndex = i
		}
	}
	if o.bestSolution == nil || bestValidationScore > o.bestSolution.Score {
		o.state.countBestScoreUnchanged = 0
		o.bestSolution = &ScoredVector{
			Vector: o.population.Members[bestIndex].TargetVector.Clone(),
			Score:  bestValidationScore,
		}
		o.onNewBest(o.bestSolution)
	} else {
		o.state.countBestScoreUnchanged++
	}
}

// resetPopulation resets the population retaining the best solution.
func (o *DifferentialEvolution) resetPopulation() {
	o.population = NewRandomPopulation(
		o.PopulationSize,
		o.VectorSize,
		o.Bound,
		rand.NewLockedRand(o.Seed),
		MemberHyperParams{
			MutationFactor: o.MutationFactor,
			CrossoverRate:  o.CrossoverRate,
			WeightFactor:   o.WeightFactor,
		},
	)
	// retain the best solution
	members := o.population.Members
	members[0].TargetVector = o.bestSolution.Vector.Clone()
	members[0].TargetScore = o.bestSolution.Score
}
