// ============================================
// OphthoAI — Client-Side Logic
// ============================================

const $ = id => document.getElementById(id);
let imageBase64 = null;
let selectedModel = 'MobileNetV2';

// ---------- Drop Zone ----------
const dropZone = $('dropZone');
const fileInput = $('fileInput');
const preview = $('imagePreview');
const dropContent = $('dropZoneContent');
const predictBtn = $('predictBtn');
const clearBtn = $('clearBtn');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', e => { if (e.target.files.length) handleFile(e.target.files[0]); });

function handleFile(file) {
    if (!file.type.startsWith('image/')) return showError('Please upload an image file.');
    const reader = new FileReader();
    reader.onload = e => {
        preview.src = e.target.result;
        preview.style.display = 'block';
        dropContent.style.display = 'none';
        imageBase64 = e.target.result.split(',')[1];
        predictBtn.disabled = false;
        clearBtn.style.display = 'flex';
    };
    reader.readAsDataURL(file);
}

clearBtn.addEventListener('click', () => {
    preview.style.display = 'none';
    dropContent.style.display = 'flex';
    imageBase64 = null;
    predictBtn.disabled = true;
    clearBtn.style.display = 'none';
    fileInput.value = '';
    $('resultsSection').style.display = 'none';
    $('loadingOverlay').style.display = 'none';
});

// ---------- Custom Select ----------
const selectTrigger = $('selectTrigger');
const selectOptions = $('selectOptions');

selectTrigger.addEventListener('click', e => {
    e.stopPropagation();
    selectOptions.classList.toggle('open');
});

document.addEventListener('click', () => selectOptions.classList.remove('open'));

document.querySelectorAll('.select-option').forEach(opt => {
    opt.addEventListener('click', () => {
        selectedModel = opt.dataset.value;
        $('selectedModel').textContent = selectedModel;
        document.querySelectorAll('.select-option').forEach(o => o.classList.remove('active'));
        opt.classList.add('active');
        selectOptions.classList.remove('open');
    });
});

// ---------- Predict ----------
predictBtn.addEventListener('click', async () => {
    if (!imageBase64) return;
    const compare = $('compareToggle').checked;
    $('loadingOverlay').style.display = 'block';
    $('resultsSection').style.display = 'none';

    try {
        if (compare) {
            const res = await fetch('/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageBase64 })
            });
            if (!res.ok) throw new Error('Server error');
            const data = await res.json();
            renderCompareResults(data);
        } else {
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: selectedModel, image: imageBase64 })
            });
            if (!res.ok) throw new Error('Server error');
            const data = await res.json();
            renderSingleResult(data);
        }
    } catch (err) {
        showError('Prediction failed. Check the server console.');
    } finally {
        $('loadingOverlay').style.display = 'none';
    }
});

// ---------- Render Single ----------
function renderSingleResult(data) {
    $('resultsSection').style.display = 'block';
    $('singleResult').style.display = 'block';
    $('compareResults').style.display = 'none';
    $('resultsDesc').textContent = `${data.model} prediction with Grad-CAM explanation`;
    $('resultModelTag').textContent = data.model;
    $('resultClass').textContent = data.class;

    // Confidence ring
    const conf = parseFloat(data.confidence);
    $('confidenceValue').textContent = conf.toFixed(1) + '%';
    const circumference = 2 * Math.PI * 52;
    const offset = circumference - (conf / 100) * circumference;
    const ring = $('ringFill');
    ring.style.strokeDasharray = circumference;
    ring.style.strokeDashoffset = circumference;
    requestAnimationFrame(() => { ring.style.strokeDashoffset = offset; });

    // Probability bars
    const barsContainer = $('probBars');
    barsContainer.innerHTML = '';
    const maxIdx = data.prob_values.indexOf(Math.max(...data.prob_values.map(Number)).toFixed(1));
    data.probs.forEach((name, i) => {
        const val = parseFloat(data.prob_values[i]);
        const isPrimary = i === parseInt(maxIdx);
        const item = document.createElement('div');
        item.className = 'prob-bar-item';
        item.innerHTML = `
            <div class="prob-bar-header">
                <span class="prob-bar-name">${name}</span>
                <span class="prob-bar-value">${val.toFixed(1)}%</span>
            </div>
            <div class="prob-bar-track">
                <div class="prob-bar-fill ${isPrimary ? '' : 'secondary'}" style="width:0"></div>
            </div>`;
        barsContainer.appendChild(item);
        requestAnimationFrame(() => {
            item.querySelector('.prob-bar-fill').style.width = val + '%';
        });
    });

    // Original image
    $('originalImg').src = preview.src;

    // Heatmap
    const heatmapImg = $('heatmapImg');
    if (data.heatmap) {
        heatmapImg.src = 'data:image/png;base64,' + data.heatmap;
        heatmapImg.parentElement.style.display = 'block';
    } else {
        heatmapImg.parentElement.style.display = 'none';
    }

    $('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------- Render Compare ----------
function renderCompareResults(data) {
    $('resultsSection').style.display = 'block';
    $('singleResult').style.display = 'none';
    $('compareResults').style.display = 'block';
    $('resultsDesc').textContent = 'Comparison across all 4 models with ensemble prediction';

    // Ensemble
    $('ensembleClass').textContent = data.ensemble.class;
    $('ensembleConfidence').textContent = parseFloat(data.ensemble.confidence).toFixed(1) + '%';

    // Model cards
    const grid = $('modelCardsGrid');
    grid.innerHTML = '';
    for (const [modelName, result] of Object.entries(data.models)) {
        const isOcp = result.class.toLowerCase().includes('ocp');
        const card = document.createElement('div');
        card.className = 'compare-card';
        let heatmapHtml = '';
        if (result.heatmap) {
            heatmapHtml = `<img class="compare-heatmap" src="data:image/png;base64,${result.heatmap}" alt="Grad-CAM for ${modelName}">`;
        }
        let probsHtml = '<div class="compare-probs">';
        result.probs.forEach((name, i) => {
            probsHtml += `<div class="compare-prob-item"><span>${name}</span><span>${parseFloat(result.prob_values[i]).toFixed(1)}%</span></div>`;
        });
        probsHtml += '</div>';

        card.innerHTML = `
            <div class="compare-card-header">
                <span class="compare-model-name">${modelName}</span>
                <span class="compare-confidence">${parseFloat(result.confidence).toFixed(1)}%</span>
            </div>
            <span class="compare-class ${isOcp ? 'ocp' : ''}">${result.class}</span>
            ${probsHtml}
            ${heatmapHtml}`;
        grid.appendChild(card);
    }

    $('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------- Error Toast ----------
function showError(msg) {
    let toast = document.querySelector('.error-toast');
    if (!toast) {
        toast = document.createElement('div');
        toast.className = 'error-toast';
        document.body.appendChild(toast);
    }
    toast.textContent = msg;
    requestAnimationFrame(() => toast.classList.add('visible'));
    setTimeout(() => toast.classList.remove('visible'), 4000);
}
