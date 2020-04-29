// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import (
	"github.com/nlpodyssey/spago/pkg/ml/ag"
	"github.com/nlpodyssey/spago/pkg/ml/optimizers/gd"
)

// TrackParamsForOptimization inserts all model params into the optimizer.
func TrackParamsForOptimization(m Model, o *gd.GradientDescent) {
	m.ForEachParam(func(param *Param) {
		o.Track(param)
	})
}

// AttachParamsToGraph inserts the params in the graph by means of wrapper nodes.
func AttachParamsToGraph(g *ag.Graph, params ...*Param) []ag.Node {
	nodes := make([]ag.Node, len(params))
	for i, p := range params {
		nodes[i] = g.NewWrap(p)
	}
	return nodes
}
