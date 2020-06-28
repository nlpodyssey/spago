// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ner

import "html/template"

const script template.JS = `
const apiUrl = '../analyze';

async function analyze() {
	try {
		enableLoadingState();
		removeEntitiesSelection();

		const text = document.getElementById('text').value;
		const options = {
			mergeEntities: document.getElementById('merge-entities').checked,
			filterNotEntities: document.getElementById('filter-not-entities').checked,
		}
		const response = await fetchEntities(text, options);

		const entities = buildEntities(response.tokens);
		showEntityButtons(entities);
		setHighlightableText(text, entities);
		
		if (entities.length === 0) {
			const took = document.createElement('div');
			took.className = 'text-gray mt-2';
			took.innerText = 'No entities found.';
			document.getElementById('entities').appendChild(took);
		}
		
		const took = document.createElement('div');
		took.className = 'text-gray mt-2';
		took.innerText = 'took ' + response.took;
		document.getElementById('entities').appendChild(took);
	} catch (e) {
		console.error(e);
		showError(e);
	}
	disableLoadingState();
}

function buildEntities(tokens) {
	const labelIds = new Map();
	return tokens.map((token, id) => {
		if (!labelIds.has(token.label)) {
			labelIds.set(token.label, labelIds.size);
		}
		return {
			...token,
			id,
			labelId: labelIds.get(token.label),
		};
	});
}

async function fetchEntities(text, options) {
	const response = await fetch(apiUrl, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ text, options })
	});
	return response.json();
}

function showEntityButtons(entities) {
	const aside = document.getElementById('entities');
	aside.innerHTML = '';

	const entitiesByLabel = new Map();
	entities.forEach((entity) => {
		entitiesByLabel.set(entity.labelId, [
			...entitiesByLabel.get(entity.labelId) || [],
			entity,
		])
	});

	entitiesByLabel.forEach((groupEntities) => showEntitiesGroup(groupEntities));
}

function showEntitiesGroup(entities) {
	const labelId = entities[0].labelId;
	const label = entities[0].label;

	const aside = document.getElementById('entities');

	const group = document.createElement('div');
	group.className = 'group';
	aside.appendChild(group);

	const btnWrapper = document.createElement('div');
	btnWrapper.className = 'label flex';
	group.appendChild(btnWrapper);

	const btn = document.createElement('button');
	btn.className = 'flex-grow font-bold p-2 cursor-pointer border-0 bg-white hover:bg-gray-200 rounded shadow flex';
	btn.classList.add('entity-label-' + labelId);
	btn.innerText = label;
	btn.onclick = selectEntity.bind(this, 'label-' + labelId);
	btnWrapper.appendChild(btn);

	entities.forEach((entity) => {
		const btnWrapper = document.createElement('div');
		btnWrapper.className = 'text flex';
		group.appendChild(btnWrapper);

		const btn = document.createElement('button');
		btn.className = 'flex-grow p-2 cursor-pointer border-0 bg-white hover:bg-gray-200 rounded shadow flex';
		btn.classList.add('entity-' + entity.id);
		btn.innerText = entity.text;
		btn.onclick = selectEntity.bind(this, entity.id);
		btnWrapper.appendChild(btn);
	});
}

function setHighlightableText(text, entities) {
	const highlightableText = document.getElementById('highlightable-text');
	highlightableText.innerHTML = '';

	let cues = createCues(entities);
	const cuesLength = cues.length;

	let curIds = [];
	let curLabelIds = [];
	let lastPos = 0;

	const makeSpan = (start, end) => {
		const t = text.substring(start, end);
		if (t.length > 0) {
			const span = document.createElement('span');
			span.innerText = t;
			span.className = curIds.map(id => 'entity-' + id)
				.concat(curLabelIds.map(id => 'entity-label-' + id))
				.join(' ');
			highlightableText.appendChild(span);
		}
	};

	for (let i = 0; i < cuesLength;) {
		const curPos = cues[i].pos;

		makeSpan(lastPos, curPos);

		for (; i < cuesLength && cues[i].pos === curPos; i++) {
			const { id, labelId, type } = cues[i];
			if (type === 'start') {
				curIds.push(id);
				curLabelIds.push(labelId);
			} else {
				curIds = curIds.filter(cid => cid !== id);
				curLabelIds = curLabelIds.filter(cid => cid !== labelId);
			}
		}

		lastPos = curPos;
	}

	makeSpan(lastPos, text.length);
}

function createCues(entities) {
	const cues = entities.reduce((arr, entity) => [
		...arr,
		{ pos: entity.start, type: 'start', id: entity.id, labelId: entity.labelId },
		{ pos: entity.end, type: 'end', id: entity.id, labelId: entity.labelId },
	], []);
	return cues.sort((a, b) => a.pos - b.pos);
}

let selectedId = null;

function selectEntity(id) { // TODO:
	const highlightableText = document.getElementById('highlightable-text');
	const alreadySelected = id === selectedId;
	removeEntitiesSelection();
	if (alreadySelected) {
		return;
	}

	selectedId = id;
	const newSpans = highlightableText.querySelectorAll('.entity-' + id);
	newSpans.forEach(s => s.classList.add('active'));

	const aside = document.getElementById('entities');
	aside.getElementsByClassName('entity-' + id)[0].classList.add('active')
}

function removeEntitiesSelection() {
	if (selectedId === null) {
		return;
	}
	selectedId = null;

	const highlightableText = document.getElementById('highlightable-text');
	const activeSpans = highlightableText.querySelectorAll('.active');
	activeSpans.forEach(s => s.classList.remove('active'));

	const aside = document.getElementById('entities');
	aside.querySelector('.active').classList.remove('active');
}

function handleTextareaInput() {
	removeEntitiesSelection();
	document.getElementById('highlightable-text').innerHTML = '';
}

function enableLoadingState() {
	document.getElementById('text').setAttribute('disabled', '');
	document.getElementById('merge-entities').setAttribute('disabled', '');
	document.getElementById('filter-not-entities').setAttribute('disabled', '');
	document.getElementById('submit').classList.add('hidden');
	document.getElementById('loader').classList.remove('hidden');
}

function disableLoadingState() {
	document.getElementById('text').removeAttribute('disabled');
	document.getElementById('merge-entities').removeAttribute('disabled');
	document.getElementById('filter-not-entities').removeAttribute('disabled');
	document.getElementById('submit').classList.remove('hidden');
	document.getElementById('loader').classList.add('hidden');
}

function showError(e) {
	const aside = document.getElementById('entities');
	aside.innerHTML = '';

	const div = document.createElement('div');
	div.className = 'text-red';
	div.innerText = e.name + '\n' + e.message;
	aside.appendChild(div);
}

async function handleTextareaScroll() {
	const highlightableText = document.getElementById('highlightable-text');
	const text = document.getElementById('text');
	highlightableText.scrollTop = text.scrollTop;
}
`
