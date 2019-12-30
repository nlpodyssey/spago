// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lstm

import (
	"brillion.io/spago/pkg/mat"
	"brillion.io/spago/pkg/ml/nn"
	"io"
)

type Model struct {
	WIn      *nn.Param `type:"weights"`
	WInRec   *nn.Param `type:"weights"`
	BIn      *nn.Param `type:"biases"`
	WOut     *nn.Param `type:"weights"`
	WOutRec  *nn.Param `type:"weights"`
	BOut     *nn.Param `type:"biases"`
	WFor     *nn.Param `type:"weights"`
	WForRec  *nn.Param `type:"weights"`
	BFor     *nn.Param `type:"biases"`
	WCand    *nn.Param `type:"weights"`
	WCandRec *nn.Param `type:"weights"`
	BCand    *nn.Param `type:"biases"`
}

func New(in, out int) *Model {
	var m Model
	m.WIn, m.WInRec, m.BIn = newGateParams(in, out)
	m.WOut, m.WOutRec, m.BOut = newGateParams(in, out)
	m.WFor, m.WForRec, m.BFor = newGateParams(in, out)
	m.WCand, m.WCandRec, m.BCand = newGateParams(in, out)
	return &m
}

func newGateParams(in, out int) (w, wRec, b *nn.Param) {
	w = nn.NewParam(mat.NewEmptyDense(out, in))
	wRec = nn.NewParam(mat.NewEmptyDense(out, out))
	b = nn.NewParam(mat.NewEmptyVecDense(out))
	return
}

func (m *Model) ForEachParam(callback func(param *nn.Param)) {
	nn.ForEachParam(m, callback)
}

func (m *Model) Serialize(w io.Writer) (int, error) {
	return nn.Serialize(m, w)
}

func (m *Model) Deserialize(r io.Reader) (int, error) {
	return nn.Deserialize(m, r)
}
