// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bls

import (
	"github.com/nlpodyssey/spago/pkg/mat"
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/nn"
	"log"
)

// BroadLearningAlgorithm performs the ridge regression approximation to optimize the output params (Wo).
// The parameters for feature mapping (Wz) can also be optimized through the alternating direction method of
// multipliers (ADMM) method (Goldstein et al. 2014).
// The parameters of the enhanced nodes remain the initial ones and are not optimized.
type BroadLearningAlgorithm[T mat.DType] struct {
	Model                  *Model[T]
	Input                  []mat.Matrix[T]
	DesiredOutput          []mat.Matrix[T]
	Penalty                T
	OptimizeFeaturesWeight bool // skip optimization if you don't want to
	Verbose                bool
}

// Do runs the board learning algorithm.
func (l *BroadLearningAlgorithm[T]) Do() {
	if l.OptimizeFeaturesWeight {
		l.log("Optimizing features weights...")
		l.optimizeFeaturesWeight()
	}
	l.log("Collecting features and enhanced nodes...")
	zh := mat.ConcatH(l.zhs()...)
	y := mat.ConcatH(l.DesiredOutput...)
	l.log("Performing ridge regression. It will take a while...")
	w := ridgeRegression[T](zh, y, l.Penalty)
	l.updateOutputWeights(w)
	l.log("All right, the model is served.")
}

func (l *BroadLearningAlgorithm[T]) optimizeFeaturesWeight() {
	featuresMap := make([][]mat.Matrix[T], l.Model.NumOfFeatures)
	for _, x := range l.Input {
		g := ag.NewGraph[T]()
		x := g.NewVariable(x, false)
		m := nn.ReifyForTraining(l.Model, g)
		for j := 0; j < m.NumOfFeatures; j++ {
			featuresMap[j] = append(featuresMap[j], nn.Affine[T](m.Graph(), m.Bz[j], m.Wz[j], x).Value())
		}
	}
	x := mat.ConcatH(l.Input...)
	for i := 0; i < l.Model.NumOfFeatures; i++ {
		z := mat.ConcatH(featuresMap[i]...)
		wz := admn[T](z, x, 1e-3, 100) // weight optimization
		l.Model.Wz[i].Value().SetData(wz.T().Data())
	}
}

func (l *BroadLearningAlgorithm[T]) zhs() []mat.Matrix[T] {
	zhs := make([]mat.Matrix[T], len(l.Input))
	for i, x := range l.Input {
		g := ag.NewGraph[T]()
		x := g.NewVariable(x, false)
		m := nn.ReifyForTraining(l.Model, g)
		zhs[i] = singleZH[T](m, x)
	}
	return zhs
}

func (l *BroadLearningAlgorithm[T]) updateOutputWeights(w mat.Matrix[T]) {
	l.Model.W.Value().SetData(w.T().Data())
}

func (l *BroadLearningAlgorithm[T]) log(message string) {
	if l.Verbose {
		log.Println(message)
	}
}

func singleZH[T mat.DType](m *Model[T], x ag.Node[T]) mat.Matrix[T] {
	g := m.Graph()
	z := m.useFeaturesDropout(m.featuresMapping(x))
	h := m.useEnhancedNodesDropout(g.Invoke(m.EnhancedNodesActivation, nn.Affine[T](g, m.Bh, m.Wh, z)))
	return g.Concat(z, h).Value()
}

// ridgeRegression obtains the solution of output weight solving W = Inv(T(A)A+λI)T(A)Y
func ridgeRegression[T mat.DType](x, y mat.Matrix[T], c T) mat.Matrix[T] {
	i2 := mat.NewIdentityDense[T](x.Columns()).ProdScalar(c)
	x2 := x.T().Mul(x).Add(i2)
	invX2 := x2.Inverse()
	return invX2.Mul(x.T()).Mul(y)
}

// admn is a naive implementation of the alternating direction method of multipliers method (Goldstein et al. 2014).
func admn[T mat.DType](z mat.Matrix[T], x mat.Matrix[T], lam T, iterations int) mat.Matrix[T] {
	ZZ := z.T().Mul(z)
	var Wk mat.Matrix[T] = mat.NewEmptyDense[T](z.Columns(), x.Columns())
	var Ok mat.Matrix[T] = mat.NewEmptyDense[T](z.Columns(), x.Columns())
	var Uk mat.Matrix[T] = mat.NewEmptyDense[T](z.Columns(), x.Columns())

	L1 := ZZ.AddInPlace(mat.NewIdentityDense[T](z.Columns()))
	L1 = L1.Inverse()
	L2 := L1.Mul(z.T()).Mul(x)

	for i := 0; i < iterations; i++ {
		temp := Ok.Sub(Uk)
		Ck := L2.Add(L1.Mul(temp))
		Ok = shrinkage(Ck.Add(Uk), lam)
		Uk = Uk.Add(Ck.Sub(Ok))
		Wk = Ok
	}
	return Wk
}

func shrinkage[T mat.DType](X mat.Matrix[T], k T) mat.Matrix[T] {
	Zeros := mat.NewEmptyDense[T](X.Rows(), X.Columns())
	X1 := X.SubScalar(k)
	X2 := X.ProdScalar(-1.0).SubScalar(k)
	X2 = X2.Maximum(Zeros)
	X2 = Zeros.Sub(X2)
	return X1.Maximum(X2)
}
