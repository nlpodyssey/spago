// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sequencelabeler

import (
	"bytes"
	"fmt"
)

// TODO: make sure that the input label sequence is valid
func mergeEntities(tokens []TokenLabel) []TokenLabel {
	newTokens := make([]TokenLabel, 0)
	buf := TokenLabel{}
	text := bytes.NewBufferString("")
	for _, token := range tokens {
		switch token.Label[0] {
		case 'O':
			newTokens = append(newTokens, token)
		case 'S':
			newToken := token
			newToken.Label = newToken.Label[2:]
			newTokens = append(newTokens, newToken)
		case 'B':
			text.Reset()
			text.Write([]byte(token.String))
			buf = TokenLabel{}
			buf.Label = fmt.Sprintf("%s", token.Label[2:]) // copy
			buf.Offsets.Start = token.Offsets.Start
		case 'I':
			text.Write([]byte(fmt.Sprintf(" %s", token.String)))
		case 'E':
			text.Write([]byte(fmt.Sprintf(" %s", token.String)))
			buf.String = text.String()
			buf.Offsets.End = token.Offsets.End
			newTokens = append(newTokens, buf)
		}
	}
	return newTokens
}

func filterNotEntities(tokens []TokenLabel) []TokenLabel {
	ret := make([]TokenLabel, 0)
	for _, token := range tokens {
		if token.Label == "O" { // not an entity
			continue
		}
		ret = append(ret, token)
	}
	return ret
}
