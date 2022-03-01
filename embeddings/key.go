// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package embeddings

import (
	"encoding/binary"
	"fmt"
)

// Key is a type constraint for the embeddings' storage keys.
type Key interface {
	[]byte | string | int
}

func encodeKey[K Key](k K) []byte {
	switch kt := any(k).(type) {
	case []byte:
		return kt
	case string:
		return []byte(kt)
	case int:
		bs := make([]byte, 8)
		binary.LittleEndian.PutUint64(bs, uint64(kt))
		return bs
	default:
		panic(fmt.Errorf("embeddings: unexpected key type %T", k))
	}
}

func stringifyKey[K Key](k K) string {
	switch kt := any(k).(type) {
	case string:
		return kt
	case []byte:
		return string(kt)
	case int:
		var arr [8]byte
		bs := arr[:]
		binary.LittleEndian.PutUint64(bs, uint64(kt))
		return string(bs)
	default:
		panic(fmt.Errorf("embeddings: unexpected key type %T", k))
	}
}
