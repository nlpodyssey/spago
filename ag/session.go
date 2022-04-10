// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ag

// ProcessingMode regulates the different usage of some operations (e.g. Dropout, BatchNorm, etc.),
// depending on whether you're doing training or inference.
// Failing to set the right mode will yield inconsistent inference results.
type ProcessingMode uint8

const (
	// Training is to be used during the training phase of a model. For example, dropouts are enabled.
	Training ProcessingMode = iota
	// Inference keeps weights fixed while using the model and disables some operations (e.g. skip dropout).
	Inference
)
