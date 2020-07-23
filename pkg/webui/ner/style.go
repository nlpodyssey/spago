// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ner

import "html/template"

const style template.CSS = `
#entities {
	width: 25rem;
}

#entities .group { margin-bottom: 1rem; }
#entities .group .label { padding: .25rem 0 .25rem 0.30rem; border-left: 0.2rem solid #555; }
#entities .group .text { padding: .25rem 0 .25rem 0.45rem; border-left: 0.05rem solid #555; }

#entities button.active,
#highlightable-text span.active {
	background-color: #8fe5fa;
}
`
