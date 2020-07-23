// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bertqa

import "html/template"

const style template.CSS = `
#answers {
	width: 25rem;
}

#answers button.active,
#highlightable-text span.active {
	background-color: #8fe5fa;
}
`
