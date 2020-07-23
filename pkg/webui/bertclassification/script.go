// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bertclassification

import "html/template"

const script template.JS = `
const apiUrl = '../classify';

async function classify() {
	try {
		enableLoadingState();
		removeClassesSelection();

		const passage = document.getElementById("passage").value;
		const response = await fetchResults(passage);

		showResults(response.distribution);
		setHighlightableText(passage, response.distribution);

		const took = document.createElement('div');
		took.className = 'text-gray mt-2';
		took.innerText = 'took ' + response.took;
		document.getElementById('classes').appendChild(took);

		if (response.distribution.length > 0) {
			selectClass(0);
		}
	} catch (e) {
		console.error(e);
		showError(e);
	}
	disableLoadingState();
}

async function fetchResults(text) {
	const response = await fetch(apiUrl, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ text }),
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
		tr.onclick = selectClass.bind(this, id);
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

function setHighlightableText(passage, distribution) {
	const highlightableText = document.getElementById('highlightable-text');
	highlightableText.innerHTML = '';

	let cues = createCues(distribution);
	const cuesLength = cues.length;

	let curIdScores = [];
	let lastPos = 0;

	const makeSpan = (start, end) => {
		const text = passage.substring(start, end);
		if (text.length > 0) {
			const span = document.createElement('span');
			span.innerText = text;
			curIdScores.forEach(({ id, score }) => {
				span.classList.add('class-' + id);
				span.dataset['scoreClass' + id] = score;
			});

			highlightableText.appendChild(span);
		}
	};

	for (let i = 0; i < cuesLength;) {
		const curPos = cues[i].pos;

		makeSpan(lastPos, curPos);

		for (; i < cuesLength && cues[i].pos === curPos; i++) {
			const { id, type, score } = cues[i];
			if (type === 'start') {
				curIdScores.push({ id, score });
			} else {
				curIdScores = curIdScores.filter(x => x.id !== id);
			}
		}

		lastPos = curPos;
	}

	makeSpan(lastPos, passage.length);
}

function createCues(answers) {
	const cues = answers.reduce(
		(arr, ans, id) => (ans.evidences || []).reduce(
			(arr, ev) => ([
				...arr,
				{ pos: ev.start, type: 'start', score: ev.score, id },
				{ pos: ev.end, type: 'end', score: ev.score, id },
			]),
			arr,
		),
		[],
	);
	return cues.sort((a, b) => a.pos - b.pos);
}

let selectedId = null;

function selectClass(id) {
	const highlightableText = document.getElementById('highlightable-text');

	const alreadySelected = id === selectedId;

	removeClassesSelection();

	if (alreadySelected) {
		return;
	}

	selectedId = id;
	const newSpans = highlightableText.querySelectorAll('.class-' + id);
	const datasetColorKey = 'scoreClass' + id;
	newSpans.forEach((s) => {
		s.classList.add('active');
		s.style.filter = 'saturate(' + s.dataset[datasetColorKey] + ')';
	});

	const aside = document.getElementById('classes');
	aside.getElementsByClassName('class-' + id)[0].classList.add('active')
}

function removeClassesSelection() {
	if (selectedId === null) {
		return;
	}
	selectedId = null;

	const highlightableText = document.getElementById('highlightable-text');
	const activeSpans = highlightableText.querySelectorAll('.active');
	activeSpans.forEach((s) => {
		s.classList.remove('active');
		s.style.filter = null;
	});

	const aside = document.getElementById('classes');
	aside.querySelector('.active').classList.remove('active');
}

function handleTextareaInput() {
	removeClassesSelection();
	document.getElementById('highlightable-text').innerHTML = '';
}

function enableLoadingState() {
	document.getElementById('passage').setAttribute('disabled', '');
	document.getElementById('submit').classList.add('hidden');
	document.getElementById('loader').classList.remove('hidden');
}

function disableLoadingState() {
	document.getElementById('passage').removeAttribute('disabled');
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

async function handleTextareaScroll() {
	const highlightableText = document.getElementById('highlightable-text');
	const passage = document.getElementById('passage');
	highlightableText.scrollTop = passage.scrollTop;
}
`
