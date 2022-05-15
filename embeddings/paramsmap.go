// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embeddings

import (
	"fmt"

	"github.com/nlpodyssey/spago/nn"
)

// ParamsMap is simply a map of embedding parameters which produces no
// data when marshaled, and expects no data when unmarshaled.
//
// This is useful to prevent cache-like data to be serialized with other
// primary Model's data. See Model.EmbeddingsWithGrad.
type ParamsMap map[string]nn.Param

// MarshalBinary satisfies encoding.BinaryMarshaler interface.
// It always produces empty data (nil) and no error.
func (pm ParamsMap) MarshalBinary() ([]byte, error) {
	return []byte{}, nil
}

// UnmarshalBinary satisfies encoding.BinaryUnmarshaler interface.
// It only accepts empty data (nil or zero-length slice), producing no
// side effects at all. If data is not blank, it returns an error.
func (pm ParamsMap) UnmarshalBinary(data []byte) error {
	if len(data) != 0 {
		return fmt.Errorf("embeddings.ParamsMap.UnmarshalBinary: empty data expected, actual data len %d", len(data))
	}
	return nil
}
