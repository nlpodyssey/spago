// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bartnli

import "html/template"

const script template.JS = `
const apiUrl = '../classify-nli';

async function classify() {
	try {
		enableLoadingState();

		const inputData = {
			text: document.getElementById("text").value,
			hypothesis_template: document.getElementById("hypothesis-template").value,
			possible_labels: document.getElementById("possible-labels").value.split(',').map(t => t.trim()).filter(t => t),
			multi_class: document.getElementById('multi-class').checked
		}

		const response = await fetchResults(inputData);

		showResults(response.distribution);

		const took = document.createElement('div');
		took.className = 'text-gray mt-2';
		took.innerText = 'took ' + response.took;
		document.getElementById('classes').appendChild(took);
	} catch (e) {
		console.error(e);
		showError(e);
	}
	disableLoadingState();
}

async function fetchResults(inputData) {
	const response = await fetch(apiUrl, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(inputData)
	});
	return response.json();
}

function showResults(distribution) {
	const aside = document.getElementById('classes');
	aside.innerHTML = '';

	const table = document.createElement('table')
	table.className = 'p-2 bg-white rounded shadow leading-none';
	table.style.width = '100%';
	table.style.borderSpacing = '0 .5rem';
	aside.appendChild(table);

	distribution.forEach((item, id) => {
		const tr = document.createElement('tr');
		tr.className = 'cursor-pointer hover:bg-gray-200';
		tr.classList.add('class-' + id);
		table.appendChild(tr);

		const tdLabel = document.createElement('td');
		tdLabel.className = 'p-1 rounded-l';
		tdLabel.innerText = item.class;
		tr.appendChild(tdLabel);

		const tdBar = document.createElement('td');
		tdBar.style.width = '100%';
		tr.appendChild(tdBar);

		const bar = document.createElement('div');
		bar.className = 'bg-blue rounded';
		bar.innerText = ' ';
		bar.style.width = (item.confidence * 100) + '%';
		bar.style.height = '1em';
		tdBar.appendChild(bar);

		const tdConfidence = document.createElement('td');
		tdConfidence.className = 'text-gray rounded-r';
		tdConfidence.innerText = item.confidence.toFixed(2);
		tr.appendChild(tdConfidence);
	});
}

function enableLoadingState() {
	document.getElementById('text').setAttribute('disabled', '');
	document.getElementById('hypothesis-template').setAttribute('disabled', '');
	document.getElementById('possible-labels').setAttribute('disabled', '');
	document.getElementById('multi-class').setAttribute('disabled', '');
	document.getElementById('submit').classList.add('hidden');
	document.getElementById('loader').classList.remove('hidden');
}

function disableLoadingState() {
	document.getElementById('text').removeAttribute('disabled');
	document.getElementById('hypothesis-template').removeAttribute('disabled');
	document.getElementById('possible-labels').removeAttribute('disabled');
	document.getElementById('multi-class').removeAttribute('disabled');
	document.getElementById('submit').classList.remove('hidden');
	document.getElementById('loader').classList.add('hidden');
}

function showError(e) {
	const aside = document.getElementById('classes');
	aside.innerHTML = '';

	const div = document.createElement('div');
	div.className = 'text-red';
	div.innerText = e.name + '\n' + e.message;
	aside.appendChild(div);
}
`
