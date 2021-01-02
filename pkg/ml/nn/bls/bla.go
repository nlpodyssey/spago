// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bls

import (
	mat "github.com/nlpodyssey/spago/pkg/mat32"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
)

// BroadLearningAlgorithm performs the ridge regression approximation to optimize the output params (Wo).
// The parameters for feature mapping (Wz) can also be optimized through the alternating direction method of
// multipliers (ADMM) method (Goldstein et al. 2014).
// The parameters of the enhanced nodes remain the initial ones and are not optimized.
type BroadLearningAlgorithm struct {
	Model                  *Model
	Input                  []mat.Matrix
	DesiredOutput          []mat.Matrix
	Penalty                mat.Float
	OptimizeFeaturesWeight bool // skip optimization if you don't want to
	Verbose                bool
}

// Do runs the board learning algorithm.
func (l *BroadLearningAlgorithm) Do() {
	if l.OptimizeFeaturesWeight {
		l.log("Optimizing features weights...")
		l.optimizeFeaturesWeight()
	}
	l.log("Collecting features and enhanced nodes...")
	zh := mat.ConcatH(l.zhs()...)
	y := mat.ConcatH(l.DesiredOutput...)
	l.log("Performing ridge regression. It will take a while...")
	w := ridgeRegression(zh, y, l.Penalty)
	l.updateOutputWeights(w)
	l.log("All right, the model is served.")
}

func (l *BroadLearningAlgorithm) optimizeFeaturesWeight() {
	featuresMap := make([][]mat.Matrix, l.Model.NumOfFeatures)
	for _, x := range l.Input {
		g := ag.NewGraph()
		x := g.NewVariable(x, false)
		m := nn.Reify(nn.Context{Graph: g, Mode: nn.Training}, l.Model).(*Model)
		for j := 0; j < m.NumOfFeatures; j++ {
			featuresMap[j] = append(featuresMap[j], nn.Affine(m.Graph(), m.Bz[j], m.Wz[j], x).Value())
		}
	}
	x := mat.ConcatH(l.Input...)
	for i := 0; i < l.Model.NumOfFeatures; i++ {
		z := mat.ConcatH(featuresMap[i]...)
		wz := admn(z, x, 1e-3, 100) // weight optimization
		l.Model.Wz[i].Value().SetData(wz.T().Data())
	}
}

func (l *BroadLearningAlgorithm) zhs() []mat.Matrix {
	zhs := make([]mat.Matrix, len(l.Input))
	for i, x := range l.Input {
		g := ag.NewGraph()
		x := g.NewVariable(x, false)
		m := nn.Reify(nn.Context{Graph: g, Mode: nn.Training}, l.Model).(*Model)
		zhs[i] = singleZH(m, x)
	}
	return zhs
}

func (l *BroadLearningAlgorithm) updateOutputWeights(w mat.Matrix) {
	l.Model.W.Value().SetData(w.T().Data())
}

func (l *BroadLearningAlgorithm) log(message string) {
	if l.Verbose {
		log.Println(message)
	}
}

func singleZH(m *Model, x ag.Node) *mat.Dense {
	g := m.Graph()
	z := m.useFeaturesDropout(m.featuresMapping(x))
	h := m.useEnhancedNodesDropout(g.Invoke(m.EnhancedNodesActivation, nn.Affine(g, m.Bh, m.Wh, z)))
	return g.Concat(z, h).Value().(*mat.Dense)
}

// ridgeRegression obtains the solution of output weight solving W = Inv(T(A)A+λI)T(A)Y
func ridgeRegression(x *mat.Dense, y *mat.Dense, c mat.Float) mat.Matrix {
	i2 := mat.I(x.Columns()).ProdScalar(c)
	x2 := x.T().Mul(x).Add(i2)
	invX2 := x2.(*mat.Dense).Inverse()
	return invX2.Mul(x.T()).Mul(y)
}

// admn is a naive implementation of the alternating direction method of multipliers method (Goldstein et al. 2014).
func admn(z *mat.Dense, x *mat.Dense, lam mat.Float, iterations int) mat.Matrix {
	ZZ := z.T().Mul(z)
	Wk := mat.NewEmptyDense(z.Columns(), x.Columns())
	Ok := mat.NewEmptyDense(z.Columns(), x.Columns())
	Uk := mat.NewEmptyDense(z.Columns(), x.Columns())

	L1 := ZZ.AddInPlace(mat.I(z.Columns()))
	L1 = L1.(*mat.Dense).Inverse()
	L2 := L1.Mul(z.T()).Mul(x)

	for i := 0; i < iterations; i++ {
		temp := Ok.Sub(Uk)
		Ck := L2.Add(L1.Mul(temp))
		Ok = shrinkage(Ck.Add(Uk).(*mat.Dense), lam).(*mat.Dense)
		Uk = Uk.Add(Ck.Sub(Ok)).(*mat.Dense)
		Wk = Ok
	}
	return Wk
}

func shrinkage(X *mat.Dense, k mat.Float) mat.Matrix {
	Zeros := mat.NewEmptyDense(X.Rows(), X.Columns())
	X1 := X.SubScalar(k).(*mat.Dense)
	X2 := X.ProdScalar(-1.0).SubScalar(k).(*mat.Dense)
	X2 = X2.Maximum(Zeros)
	X2 = Zeros.Sub(X2).(*mat.Dense)
	return X1.Maximum(X2)
}
