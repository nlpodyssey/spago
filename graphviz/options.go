// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package graphviz

// Options allows customization of generated graphviz graphs.
type Options struct {
	// ColoredTimeSteps indicates whether to use different colors for
	// representing nodes with different time-step values.
	ColoredTimeSteps bool
	// ShowNodesWithoutEdges indicates whether to show graph nodes
	// which have no connections.
	ShowNodesWithoutEdges bool
}
