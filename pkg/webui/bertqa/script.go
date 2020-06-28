// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bertqa

import "html/template"

const script template.JS = `
const apiUrl = '../answer';

async function answer() {
	try {
		enableLoadingState();
		removeAnswerSelection();

		const passage = document.getElementById("passage").value;
		const question = document.getElementById("question").value;
		const response = await fetchAnswer(passage, question);

		showAnswerButtons(response.answers);
		setHighlightableText(passage, response.answers);

		const took = document.createElement('div');
		took.className = 'text-gray mt-2';
		took.innerText = 'took ' + response.took;
		document.getElementById('answers').appendChild(took);

		if (response.answers.length > 0) {
			selectAnswer(0);
		}
	} catch (e) {
		console.error(e);
		showError(e);
	}
	disableLoadingState();
}

async function fetchAnswer(passage, question) {
	const response = await fetch(apiUrl, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ passage, question })
	});
	return response.json();
}

function showAnswerButtons(answers) {
	const aside = document.getElementById('answers');
	aside.innerHTML = '';
	answers.forEach((answer, id) => {
		const btn = document.createElement('button');
		btn.className = 'mb-2 p-2 cursor-pointer border-0 bg-white hover:bg-gray-200 rounded shadow flex';
		btn.classList.add('answer-' + id);
		btn.onclick = selectAnswer.bind(this, id);
		aside.appendChild(btn);

		const text = document.createElement('span');
		text.className = 'flex-grow';
		text.innerText = answer.text;
		btn.appendChild(text)

		const confidence = document.createElement('span');
		confidence.className = 'text-gray';
		confidence.innerText = answer.confidence.toFixed(2);
		btn.appendChild(confidence)
	});
}

function setHighlightableText(passage, answers) {
	const highlightableText = document.getElementById('highlightable-text');
	highlightableText.innerHTML = '';

	let cues = createCues(answers);
	const cuesLength = cues.length;

	let curIds = [];
	let lastPos = 0;

	const makeSpan = (start, end) => {
		const text = passage.substring(start, end);
		if (text.length > 0) {
			const span = document.createElement('span');
			span.innerText = text;
			span.className = curIds.map(id => 'answer-' + id).join(' ');
			highlightableText.appendChild(span);
		}
	};

	for (let i = 0; i < cuesLength;) {
		const curPos = cues[i].pos;

		makeSpan(lastPos, curPos);

		for (; i < cuesLength && cues[i].pos === curPos; i++) {
			const { id, type } = cues[i];
			if (type === 'start') {
				curIds.push(id);
			} else {
				curIds = curIds.filter(cid => cid !== id);
			}
		}

		lastPos = curPos;
	}

	makeSpan(lastPos, passage.length);
}

function createCues(answers) {
	const cues = answers.reduce((arr, ans, id) => [
		...arr,
		{ pos: ans.start, type: 'start', id },
		{ pos: ans.end, type: 'end', id },
	], []);
	return cues.sort((a, b) => a.pos - b.pos);
}

let selectedId = null;

function selectAnswer(id) {
	const highlightableText = document.getElementById('highlightable-text');

	const alreadySelected = id === selectedId;

	removeAnswerSelection();

	if (alreadySelected) {
		return;
	}

	selectedId = id;
	const newSpans = highlightableText.querySelectorAll('.answer-' + id);
	newSpans.forEach(s => s.classList.add('active'));

	const aside = document.getElementById('answers');
	aside.getElementsByClassName('answer-' + id)[0].classList.add('active')
}

function removeAnswerSelection() {
	if (selectedId === null) {
		return;
	}
	selectedId = null;

	const highlightableText = document.getElementById('highlightable-text');
	const activeSpans = highlightableText.querySelectorAll('.active');
	activeSpans.forEach(s => s.classList.remove('active'));

	const aside = document.getElementById('answers');
	aside.querySelector('.active').classList.remove('active');
}

function handleTextareaInput() {
	removeAnswerSelection();
	document.getElementById('highlightable-text').innerHTML = '';
}

function enableLoadingState() {
	document.getElementById('passage').setAttribute('disabled', '');
	document.getElementById('question').setAttribute('disabled', '');
	document.getElementById('submit').classList.add('hidden');
	document.getElementById('loader').classList.remove('hidden');
}

function disableLoadingState() {
	document.getElementById('passage').removeAttribute('disabled');
	document.getElementById('question').removeAttribute('disabled');
	document.getElementById('submit').classList.remove('hidden');
	document.getElementById('loader').classList.add('hidden');
}

function showError(e) {
	const aside = document.getElementById('answers');
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
