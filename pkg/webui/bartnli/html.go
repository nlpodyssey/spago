// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bartnli

import (
	"bytes"
	"github.com/nlpodyssey/spago/pkg/webui"
	"html/template"
	"log"
	"net/http"
)

const htmlTemplate = `
<!doctype html>
<html lang="">

<head>
	<meta charset="utf-8">
	<title>spaGO</title>
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<style>
		{{.CommonStyle}}
		{{.CustomStyle}}
	</style>
</head>

<body class="flex flex-col">
	<header class="p-2 shadow z-10">
		<a
			class="text-xl font-bold italic"
			href="https://github.com/nlpodyssey/spago"
			target="_blank"
		>
			spa<em class="text-blue">GO</em>
		</a>
	</header>

	<main class="flex-grow overflow-hidden flex">
		<form
			class="bg-gray-200 flex-grow p-4 flex flex-col"
			onsubmit="classify(); return false;"
		>
			<label for="text" class="text-sm">Text</label>
			<div class="flex-grow max-h-96 bg-white rounded shadow flex overflow-hidden relative">
				<textarea
					id="text"
					placeholder="Text..."
					class="flex-grow resize-none bg-transparent p-2 z-10 overflow-auto"
				></textarea>
			</div>

			<label for="hypothesis-template" class="text-sm mt-4">Hypothesis template (use "{}" as label placeholder)</label>
			<input
				type="text"
				id="hypothesis-template"
				placeholder="Hypothesis template..."
				class="p-2 rounded shadow"
				value="This text is about {}"
			>

			<label for="possible-labels" class="text-sm mt-4">Possible labels (separated by comma ",")</label>
			<input
				type="text"
				id="possible-labels"
				placeholder="Possible labels..."
				class="p-2 rounded shadow"
			>

			<div class="mt-4 flex">
				<div class="flex-grow py-2">
					<input id="multi-class" type="checkbox" checked>
					<label for="multi-class" class="mr-2">Multi class</label>
				</div>
				<input
					type="submit"
					id="submit"
					value="Classify"
					class="rounded shadow cursor-pointer py-2 px-4 bg-blue hover:bg-light-blue"
				>
				<div id="loader" class="hidden"></div>
			</div>
		</form>

		<aside id="classes" class="bg-gray-300 shadow p-4 overflow-auto flex flex-col">
		</aside>
	</main>
	<script>
		{{.Script}}
  	</script>
</body>

</html>
`

var html []byte

func init() {
	t, err := template.New("BERT Classification Web UI").Parse(htmlTemplate)
	if err != nil {
		log.Fatal(err)
	}

	data := struct {
		CommonStyle template.CSS
		CustomStyle template.CSS
		Script      template.JS
	}{
		CommonStyle: webui.CommonStyle,
		CustomStyle: style,
		Script:      script,
	}

	buf := bytes.NewBuffer([]byte{})
	err = t.Execute(buf, data)
	if err != nil {
		log.Fatal(err)
	}
	html = buf.Bytes()
}

// Handler is the server handler function for BART NLI classification web UI
func Handler(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*") // that's intended for testing purposes only
	w.Header().Set("Content-Type", "text/html")
	_, err := w.Write(html)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}
