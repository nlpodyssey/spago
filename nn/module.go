// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nn

import "encoding/gob"

var _ Model = &Module{}

// Module must be embedded into all neural models.
type Module struct{}

func init() {
	gob.Register(&Module{})
}

func (m Module) mustEmbedModule() {}
