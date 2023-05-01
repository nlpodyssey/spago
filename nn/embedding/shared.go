// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embedding

import (
	"encoding/gob"
	"fmt"
)

// Shared wraps Model, overriding binary marshaling methods in order
// to produce (and expect) empty data.
// This is useful e.g. to share embeddings between encoder and decoder models.
type Shared struct {
	*Model
}

// MarshalBinary satisfies encoding.BinaryMarshaler interface.
// It always produces empty data (nil) and no error.
func (Shared) MarshalBinary() ([]byte, error) {
	return nil, nil
}

// UnmarshalBinary satisfies encoding.BinaryUnmarshaler interface.
// It only accepts empty data (nil or zero-length slice), producing no
// side effects at all. If data is not blank, it returns an error.
func (Shared) UnmarshalBinary(data []byte) error {
	if len(data) != 0 {
		return fmt.Errorf("Shared.UnmarshalBinary: empty data expected, actual data len %d", len(data))
	}
	return nil
}

func init() {
	gob.Register(Shared{})
}
