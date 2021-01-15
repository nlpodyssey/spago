// Copyright 2019 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syncmap

import "sync"

// Map extends sync.Map preventing its binary serialization.
type Map struct {
	*sync.Map
}

// New returns a new empty Map.
func New() *Map {
	return &Map{
		Map: &sync.Map{},
	}
}

// MarshalBinary prevents Map to be encoded to binary representation.
func (Map) MarshalBinary() ([]byte, error) {
	return nil, nil
}

// UnmarshalBinary prevents Map to be decoded from binary representation.
func (m *Map) UnmarshalBinary([]byte) error {
	m.Map = &sync.Map{}
	return nil
}
