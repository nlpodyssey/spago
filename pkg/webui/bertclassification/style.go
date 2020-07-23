// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bertclassification

import "html/template"

const style template.CSS = `
#classes {
	width: 35rem;
	max-width: 50%;
}
#classes tr.active {
	background-color: #dbf6ff;
}
#highlightable-text span.active {
	background-color: #8fe5fa;
}
`
