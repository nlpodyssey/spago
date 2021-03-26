// Copyright (c) 2020 Vikesh Raj C. All rights reserved.
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

package sentencepiece

// Token holds a unit of a tokenized word
type Token struct {
	ID   int32
	Text string
}

type tokenOffset struct {
	id    int32
	text  string
	start int
	end   int
}
