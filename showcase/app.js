/* ============================================================
   Offroad Segmentation Showcase — App Logic
   Loads from data.js (REAL_DATA) when available
   ============================================================ */

// ---- Resolve data source ----
const USE_REAL = (typeof REAL_DATA !== 'undefined') && REAL_DATA.useRealData;

const CLASS_NAMES = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Flowers', 'Logs', 'Rocks', 'Landscape', 'Sky'
];

const CLASS_COLORS = [
    [0, 0, 0],        [34, 139, 34],    [0, 255, 0],
    [210, 180, 140],  [139, 90, 43],    [128, 128, 0],
    [255, 105, 180],  [139, 69, 19],    [128, 128, 128],
    [160, 82, 45],    [135, 206, 235]
];

const CLASS_CSS = CLASS_COLORS.map(c => `rgb(${c[0]},${c[1]},${c[2]})`);

// ---- Get active data ----
const ACTIVE_CLASS_IOU = USE_REAL ? REAL_DATA.classIoU : [0.12, 0.51, 0.42, 0.55, 0.38, 0.19, 0.08, 0.22, 0.41, 0.68, 0.92];

const ACTIVE_HISTORY = USE_REAL ? REAL_DATA.trainingHistory : (() => {
    const h = { epochs: 30, train_loss: [], val_loss: [], train_iou: [], val_iou: [],
                train_dice: [], val_dice: [], train_pixel_acc: [], val_pixel_acc: [], lr: [] };
    for (let i = 0; i < 30; i++) {
        const p = i / 29; const n = () => (Math.random() - 0.5) * 0.03;
        h.train_loss.push(1.8 * Math.exp(-2.5 * p) + 0.15 + n());
        h.val_loss.push(1.8 * Math.exp(-2.2 * p) + 0.25 + n() * 1.5);
        h.train_iou.push(Math.min(0.55, 0.08 + 0.47 * (1 - Math.exp(-3 * p))) + n());
        h.val_iou.push(Math.min(0.48, 0.05 + 0.43 * (1 - Math.exp(-2.5 * p))) + n());
        h.train_dice.push(Math.min(0.65, 0.1 + 0.55 * (1 - Math.exp(-3 * p))) + n());
        h.val_dice.push(Math.min(0.59, 0.08 + 0.51 * (1 - Math.exp(-2.5 * p))) + n());
        h.train_pixel_acc.push(Math.min(0.93, 0.5 + 0.43 * (1 - Math.exp(-4 * p))) + n());
        h.val_pixel_acc.push(Math.min(0.88, 0.45 + 0.43 * (1 - Math.exp(-3.5 * p))) + n());
        h.lr.push(0.001 * (0.5 * (1 + Math.cos(Math.PI * p))));
    }
    return h;
})();

const ACTIVE_CONFUSION = (USE_REAL && REAL_DATA.confusionMatrix) ? REAL_DATA.confusionMatrix : generateConfusionFromIoU(ACTIVE_CLASS_IOU);
const ACTIVE_FAILURES = USE_REAL ? REAL_DATA.failureAnalysis : [
    { cls: 'Flowers', iou: 0.08, confused: 'Lush Bushes', pct: 34, reason: 'Small, sparse objects' },
    { cls: 'Background', iou: 0.12, confused: 'Landscape', pct: 62, reason: 'No true background in dataset' },
    { cls: 'Ground Clutter', iou: 0.19, confused: 'Landscape', pct: 45, reason: 'Subtle texture differences' },
    { cls: 'Logs', iou: 0.22, confused: 'Rocks', pct: 28, reason: 'Similar color and shape' },
];

const ACTIVE_GALLERY = USE_REAL && REAL_DATA.predictionImages.length > 0
    ? REAL_DATA.predictionImages
    : generateDemoGallery(9);

const ACTIVE_CONFIG = USE_REAL ? REAL_DATA.modelConfig : {
    backbone: 'DINOv2 ViT-S/14', head: 'Enhanced Decoder', epochs: 30
};

function generateConfusionFromIoU(classIoU) {
    const n = classIoU.length;
    const cm = [];
    for (let i = 0; i < n; i++) {
        const row = [];
        const diagVal = Math.max(0.3, classIoU[i] * 0.95 + 0.05);
        let remaining = 1 - diagVal;
        for (let j = 0; j < n; j++) {
            if (i === j) {
                row.push(diagVal);
            } else {
                const offdiag = remaining * (Math.random() * 0.3 + 0.02);
                remaining -= offdiag;
                if (remaining < 0) remaining = 0;
                row.push(offdiag);
            }
        }
        const sum = row.reduce((a, b) => a + b, 0);
        cm.push(row.map(v => v / sum));
    }
    return cm;
}

function generateDemoGallery(count) {
    const items = [];
    const baseIoU = USE_REAL ? REAL_DATA.heroStats.meanIoU : 0.45;
    for (let i = 0; i < count; i++) {
        const iou = Math.max(0.15, baseIoU - 0.25 + Math.random() * 0.5);
        items.push({
            name: `test_${String(i).padStart(4, '0')}.png`,
            iou: iou,
            quality: iou > 0.7 ? 'good' : iou > 0.45 ? 'mid' : 'bad',
            isReal: false
        });
    }
    return items.sort((a, b) => b.iou - a.iou);
}

console.log(`%c📊 Data source: ${USE_REAL ? 'REAL MODEL (IoU: ' + REAL_DATA.heroStats.meanIoU + ')' : 'DEMO DATA'}`, 
    'color: #f97316; font-size: 14px; font-weight: bold;');


// ---- Procedural Desert Scene Generator ----

function mulberry32(a) {
    return function() {
        a |= 0; a = a + 0x6D2B79F5 | 0;
        var t = Math.imul(a ^ a >>> 15, 1 | a);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

function drawDesertScene(ctx, w, h, seed) {
    const rng = mulberry32(seed); const r = () => rng();
    const skyGrad = ctx.createLinearGradient(0, 0, 0, h * 0.4);
    skyGrad.addColorStop(0, `hsl(${200 + r()*30}, ${60+r()*20}%, ${70+r()*15}%)`);
    skyGrad.addColorStop(1, `hsl(${30 + r()*20}, ${40+r()*30}%, ${80+r()*10}%)`);
    ctx.fillStyle = skyGrad; ctx.fillRect(0, 0, w, h * 0.45);
    const horizonY = h * (0.35 + r() * 0.1);
    const groundGrad = ctx.createLinearGradient(0, horizonY, 0, h);
    groundGrad.addColorStop(0, `hsl(${30+r()*15}, ${40+r()*20}%, ${55+r()*15}%)`);
    groundGrad.addColorStop(0.5, `hsl(${25+r()*10}, ${45+r()*15}%, ${45+r()*10}%)`);
    groundGrad.addColorStop(1, `hsl(${20+r()*10}, ${35+r()*15}%, ${35+r()*10}%)`);
    ctx.fillStyle = groundGrad; ctx.fillRect(0, horizonY, w, h);
    ctx.fillStyle = `hsla(${25+r()*15}, ${30+r()*20}%, ${50+r()*15}%, 0.7)`;
    ctx.beginPath(); ctx.moveTo(0, horizonY + 20);
    for (let x = 0; x <= w; x += 5) ctx.lineTo(x, horizonY + Math.sin(x*0.008+r()*10)*25 + Math.sin(x*0.015+r()*5)*15);
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.fill();
    for (let i = 0; i < 5+r()*8; i++) {
        ctx.fillStyle = `hsl(${20+r()*20}, ${15+r()*15}%, ${40+r()*25}%)`;
        ctx.beginPath(); ctx.ellipse(r()*w, horizonY+30+r()*(h-horizonY-60), 15+r()*40, 10+r()*25, r()*0.3, 0, Math.PI*2); ctx.fill();
    }
    for (let i = 0; i < 8+r()*12; i++) {
        const bx = r()*w, by = horizonY+20+r()*(h-horizonY-50), bs = 8+r()*25;
        ctx.fillStyle = r()>0.5 ? `hsl(${90+r()*40},${40+r()*30}%,${25+r()*20}%)` : `hsl(${30+r()*20},${30+r()*20}%,${35+r()*15}%)`;
        for (let j=0; j<5; j++) { ctx.beginPath(); ctx.arc(bx+(r()-0.5)*bs,by+(r()-0.5)*bs*0.5,bs*0.3+r()*bs*0.3,0,Math.PI*2); ctx.fill(); }
    }
    for (let i = 0; i < 2+r()*5; i++) {
        const tx=r()*w, ty=horizonY+10+r()*(h*0.3), ts=20+r()*40;
        ctx.fillStyle = `hsl(${25+r()*10},${40+r()*15}%,${25+r()*10}%)`; ctx.fillRect(tx-3,ty,6,ts*0.8);
        ctx.fillStyle = `hsl(${100+r()*30},${35+r()*25}%,${25+r()*15}%)`; ctx.beginPath(); ctx.arc(tx,ty,ts*0.4,0,Math.PI*2); ctx.fill();
    }
}

function drawSegmentationMask(ctx, w, h, seed) {
    const rng = mulberry32(seed); const r = () => rng();
    const horizonY = h * (0.35 + r() * 0.1);
    ctx.fillStyle = CLASS_CSS[10]; ctx.fillRect(0, 0, w, horizonY+10);
    ctx.fillStyle = CLASS_CSS[9]; ctx.fillRect(0, horizonY-5, w, h);
    for (const cls of [3,8,5]) {
        for (let i=0;i<3+r()*5;i++) {
            ctx.fillStyle = CLASS_CSS[cls]; ctx.beginPath();
            ctx.ellipse(r()*w, horizonY+30+r()*(h-horizonY-50), 20+r()*50, 10+r()*25, r(), 0, Math.PI*2); ctx.fill();
        }
    }
    for (let i=0;i<5+r()*8;i++) {
        ctx.fillStyle = CLASS_CSS[8]; ctx.beginPath();
        ctx.ellipse(r()*w, horizonY+30+r()*(h-horizonY-60), 15+r()*40, 10+r()*25, r()*0.3, 0, Math.PI*2); ctx.fill();
    }
    for (let i=0;i<8+r()*12;i++) {
        const bx=r()*w,by=horizonY+20+r()*(h-horizonY-50),bs=8+r()*25,cls=r()>0.5?2:4;
        ctx.fillStyle = CLASS_CSS[cls];
        for (let j=0;j<5;j++){ctx.beginPath();ctx.arc(bx+(r()-0.5)*bs,by+(r()-0.5)*bs*0.5,bs*0.3+r()*bs*0.3,0,Math.PI*2);ctx.fill();}
    }
    for (let i=0;i<2+r()*5;i++){
        const tx=r()*w,ty=horizonY+10+r()*(h*0.3),ts=20+r()*40;
        ctx.fillStyle=CLASS_CSS[7]; ctx.fillRect(tx-3,ty,6,ts*0.8);
        ctx.fillStyle=CLASS_CSS[1]; ctx.beginPath(); ctx.arc(tx,ty,ts*0.4,0,Math.PI*2); ctx.fill();
    }
}

// ---- Chart.js ----

const chartDefaults = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color: '#94a3b8', font: { family: 'Inter', size: 12 }, boxWidth: 12, padding: 16 } } },
    scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748b', font: { family: 'Inter', size: 11 } } },
        y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748b', font: { family: 'Inter', size: 11 } } }
    }
};

function createLineChart(id, labels, datasets) {
    return new Chart(document.getElementById(id), {
        type: 'line', data: { labels, datasets },
        options: { ...chartDefaults, elements: { point: { radius: 2, hoverRadius: 5 }, line: { tension: 0.3, borderWidth: 2.5 } }, interaction: { mode: 'index', intersect: false } }
    });
}

function initCharts() {
    const h = ACTIVE_HISTORY;
    const epochs = h.train_loss.map((_, i) => i + 1);

    createLineChart('lossChart', epochs, [
        { label: 'Train Loss', data: h.train_loss, borderColor: '#f97316', backgroundColor: 'rgba(249,115,22,0.1)', fill: true },
        { label: 'Val Loss', data: h.val_loss, borderColor: '#3b82f6', backgroundColor: 'rgba(59,130,246,0.1)', fill: true }
    ]);
    createLineChart('iouChart', epochs, [
        { label: 'Train IoU', data: h.train_iou, borderColor: '#4ade80', backgroundColor: 'rgba(74,222,128,0.1)', fill: true },
        { label: 'Val IoU', data: h.val_iou, borderColor: '#a78bfa', backgroundColor: 'rgba(167,139,250,0.1)', fill: true }
    ]);
    createLineChart('diceChart', epochs, [
        { label: 'Train Dice', data: h.train_dice, borderColor: '#f472b6', backgroundColor: 'rgba(244,114,182,0.1)', fill: true },
        { label: 'Val Dice', data: h.val_dice, borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.1)', fill: true }
    ]);
    createLineChart('lrChart', epochs, [
        { label: 'Learning Rate', data: h.lr, borderColor: '#fbbf24', backgroundColor: 'rgba(251,191,36,0.1)', fill: true }
    ]);

    new Chart(document.getElementById('classIoUChart'), {
        type: 'bar',
        data: {
            labels: CLASS_NAMES,
            datasets: [{ label: 'IoU', data: ACTIVE_CLASS_IOU,
                backgroundColor: CLASS_CSS.map(c => c.replace('rgb','rgba').replace(')',',0.7)')),
                borderColor: CLASS_CSS, borderWidth: 1.5, borderRadius: 6 }]
        },
        options: { ...chartDefaults,
            scales: { ...chartDefaults.scales, y: { ...chartDefaults.scales.y, min: 0, max: 1 },
                x: { ...chartDefaults.scales.x, ticks: { ...chartDefaults.scales.x.ticks, maxRotation: 45 } } }
        }
    });
}

function mean(arr) { return arr.reduce((a, b) => a + b, 0) / arr.length; }

// ---- Class Grid ----

function initClassGrid() {
    const grid = document.getElementById('classGrid');
    CLASS_NAMES.forEach((name, i) => {
        const iou = ACTIVE_CLASS_IOU[i];
        const color = CLASS_CSS[i];
        const card = document.createElement('div');
        card.className = 'class-card reveal';
        card.style.setProperty('--class-color', color);
        card.innerHTML = `
            <div class="class-header">
                <div class="class-swatch" style="background:${color}"></div>
                <div class="class-name">${name}</div>
            </div>
            <div class="class-iou-value" style="color:${iou > 0.7 ? '#4ade80' : iou > 0.4 ? '#fbbf24' : '#f87171'}">${iou.toFixed(3)}</div>
            <div class="class-iou-bar">
                <div class="class-iou-bar-fill" style="background:${color}; --target:${iou*100}%" data-target="${iou*100}"></div>
            </div>
        `;
        grid.appendChild(card);
    });
}

// ---- Gallery ----

function initGallery() {
    const grid = document.getElementById('galleryGrid');
    ACTIVE_GALLERY.forEach((item, idx) => {
        const div = document.createElement('div');
        const isReal = item.isReal !== false && item.input;
        const iouClass = item.iou > 0.7 ? 'iou-good' : item.iou > 0.45 ? 'iou-mid' : 'iou-bad';
        div.className = 'gallery-item reveal';
        div.dataset.quality = item.quality || (item.iou > 0.7 ? 'good' : item.iou > 0.45 ? 'mid' : 'bad');

        if (isReal) {
            // Real prediction images
            div.innerHTML = `
                <div class="gallery-canvas-wrap" style="display:grid; grid-template-columns:1fr 1fr; position:relative;">
                    <img src="${item.input}" alt="Input" style="width:100%;height:100%;object-fit:cover;">
                    <img src="${item.pred}" alt="Prediction" style="width:100%;height:100%;object-fit:cover;">
                    <div style="position:absolute;top:4px;left:4px;background:rgba(0,0,0,0.6);color:white;font-size:10px;padding:2px 8px;border-radius:4px;">Input</div>
                    <div style="position:absolute;top:4px;right:4px;background:rgba(0,0,0,0.6);color:white;font-size:10px;padding:2px 8px;border-radius:4px;">Prediction</div>
                </div>
                <div class="gallery-info">
                    <span class="gallery-name">${item.name}</span>
                    <span class="gallery-iou ${iouClass}">IoU: ${item.iou.toFixed(3)}</span>
                </div>
            `;
        } else {
            // Procedural demo scene
            div.innerHTML = `
                <div class="gallery-canvas-wrap">
                    <canvas id="gallery-canvas-${idx}" width="480" height="270"></canvas>
                </div>
                <div class="gallery-info">
                    <span class="gallery-name">${item.name}</span>
                    <span class="gallery-iou ${iouClass}">IoU: ${item.iou.toFixed(3)}</span>
                </div>
            `;
            requestAnimationFrame(() => {
                const canvas = document.getElementById(`gallery-canvas-${idx}`);
                if (!canvas) return;
                const ctx = canvas.getContext('2d');
                const cw = canvas.width, ch = canvas.height, seed = 42 + idx * 137;
                ctx.save(); ctx.beginPath(); ctx.rect(0,0,cw/2,ch); ctx.clip(); drawDesertScene(ctx,cw,ch,seed); ctx.restore();
                ctx.save(); ctx.beginPath(); ctx.rect(cw/2,0,cw/2,ch); ctx.clip(); drawSegmentationMask(ctx,cw,ch,seed); ctx.restore();
                ctx.strokeStyle='rgba(255,255,255,0.5)'; ctx.lineWidth=2; ctx.setLineDash([4,4]);
                ctx.beginPath(); ctx.moveTo(cw/2,0); ctx.lineTo(cw/2,ch); ctx.stroke(); ctx.setLineDash([]);
                ctx.font='10px Inter,sans-serif';
                ctx.fillStyle='rgba(0,0,0,0.6)'; ctx.fillRect(4,4,42,16); ctx.fillRect(cw/2+4,4,72,16);
                ctx.fillStyle='white'; ctx.fillText('Input',8,16); ctx.fillText('Prediction',cw/2+8,16);
            });
        }
        grid.appendChild(div);
    });

    document.querySelectorAll('.gallery-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.gallery-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const f = btn.dataset.filter;
            document.querySelectorAll('.gallery-item').forEach(item => {
                item.style.display = f==='all' ? '' : (item.dataset.quality===f || (f==='bad'&&item.dataset.quality==='mid')) ? '' : 'none';
            });
        });
    });
}

// ---- Comparison Slider ----

function initComparisonSlider() {
    const canvas = document.getElementById('comparisonCanvas');
    const ctx = canvas.getContext('2d');
    const container = document.getElementById('comparisonSlider');
    const line = document.getElementById('sliderLine');
    const handle = document.getElementById('sliderHandle');
    const cw = canvas.width, ch = canvas.height, seed = 12345;
    let sliderPos = 0.5;

    // Check if we have real images for comparison
    const hasRealImages = USE_REAL && REAL_DATA.predictionImages.length > 0;

    if (hasRealImages) {
        const first = REAL_DATA.predictionImages[0];
        const inputImg = new Image();
        const predImg = new Image();
        let loaded = 0;

        function drawRealComparison() {
            ctx.clearRect(0, 0, cw, ch);
            const splitX = Math.floor(sliderPos * cw);
            ctx.save(); ctx.beginPath(); ctx.rect(0,0,splitX,ch); ctx.clip();
            ctx.drawImage(inputImg, 0, 0, cw, ch); ctx.restore();
            ctx.save(); ctx.beginPath(); ctx.rect(splitX,0,cw-splitX,ch); ctx.clip();
            ctx.drawImage(predImg, 0, 0, cw, ch); ctx.restore();
            line.style.left = sliderPos*100+'%';
            handle.style.left = sliderPos*100+'%';
        }

        inputImg.onload = predImg.onload = () => { if (++loaded === 2) drawRealComparison(); };
        inputImg.src = first.input;
        predImg.src = first.pred;

        let dragging = false;
        function updatePos(clientX) {
            const rect = container.getBoundingClientRect();
            sliderPos = Math.max(0.05, Math.min(0.95, (clientX - rect.left) / rect.width));
            drawRealComparison();
        }
        container.addEventListener('mousedown', e => { dragging=true; updatePos(e.clientX); });
        window.addEventListener('mousemove', e => { if(dragging) updatePos(e.clientX); });
        window.addEventListener('mouseup', () => { dragging=false; });
        container.addEventListener('touchstart', e => { dragging=true; updatePos(e.touches[0].clientX); });
        window.addEventListener('touchmove', e => { if(dragging) updatePos(e.touches[0].clientX); });
        window.addEventListener('touchend', () => { dragging=false; });
    } else {
        function drawComparison() {
            ctx.clearRect(0,0,cw,ch);
            const splitX = Math.floor(sliderPos * cw);
            ctx.save(); ctx.beginPath(); ctx.rect(0,0,splitX,ch); ctx.clip(); drawDesertScene(ctx,cw,ch,seed); ctx.restore();
            ctx.save(); ctx.beginPath(); ctx.rect(splitX,0,cw-splitX,ch); ctx.clip(); drawSegmentationMask(ctx,cw,ch,seed); ctx.restore();
            line.style.left = sliderPos*100+'%';
            handle.style.left = sliderPos*100+'%';
        }
        drawComparison();
        let dragging = false;
        function updatePos(clientX) {
            const rect = container.getBoundingClientRect();
            sliderPos = Math.max(0.05, Math.min(0.95, (clientX - rect.left) / rect.width));
            drawComparison();
        }
        container.addEventListener('mousedown', e => { dragging=true; updatePos(e.clientX); });
        window.addEventListener('mousemove', e => { if(dragging) updatePos(e.clientX); });
        window.addEventListener('mouseup', () => { dragging=false; });
        container.addEventListener('touchstart', e => { dragging=true; updatePos(e.touches[0].clientX); });
        window.addEventListener('touchmove', e => { if(dragging) updatePos(e.touches[0].clientX); });
        window.addEventListener('touchend', () => { dragging=false; });
    }
}

// ---- Confusion Matrix ----

function initConfusionMatrix() {
    const grid = document.getElementById('confusionGrid');
    const n = CLASS_NAMES.length;
    grid.style.gridTemplateColumns = `60px repeat(${n}, 1fr)`;
    grid.innerHTML = '<div></div>';
    for (let j = 0; j < n; j++) grid.innerHTML += `<div class="confusion-header confusion-header-rotated">${CLASS_NAMES[j]}</div>`;
    for (let i = 0; i < n; i++) {
        grid.innerHTML += `<div class="confusion-header" style="text-align:right;padding-right:8px;">${CLASS_NAMES[i]}</div>`;
        for (let j = 0; j < n; j++) {
            const val = ACTIVE_CONFUSION[i][j];
            const isDiag = i === j;
            const bgColor = isDiag ? `rgba(74,222,128,${val*0.8})` : `rgba(249,115,22,${val*3})`;
            const textColor = val > 0.3 ? 'white' : '#94a3b8';
            grid.innerHTML += `<div class="confusion-cell" style="background:${bgColor};color:${textColor};" title="${CLASS_NAMES[i]} → ${CLASS_NAMES[j]}: ${(val*100).toFixed(1)}%">${val > 0.02 ? (val*100).toFixed(0)+'%' : ''}</div>`;
        }
    }
}

// ---- Failure Analysis ----

function initFailureAnalysis() {
    const container = document.getElementById('failureCards');
    ACTIVE_FAILURES.forEach(f => {
        container.innerHTML += `
            <div class="failure-card">
                <h4>⚠️ ${f.cls} <span style="font-weight:400;color:var(--text-muted)">(IoU: ${f.iou.toFixed(2)})</span></h4>
                <p>Most confused with <strong>${f.confused}</strong> (<span class="failure-pct">${f.pct}%</span> of errors)</p>
                <p style="margin-top:0.5rem;color:var(--text-muted);font-size:0.8rem;">${f.reason}</p>
            </div>
        `;
    });
}

// ---- Animated Stats ----

function animateStats() {
    // Override stat values from REAL_DATA if available
    if (USE_REAL) {
        const stats = REAL_DATA.heroStats;
        const el = (id, val, suffix='') => {
            const e = document.querySelector(`#${id} .stat-value`);
            if (e) { e.dataset.target = val; e.dataset.suffix = suffix; }
            const bar = document.querySelector(`#${id} .stat-bar-fill`);
            if (bar) bar.style.setProperty('--target-width', (val * (suffix ? 1 : 100)) + (suffix ? '%' : '%'));
        };
        el('stat-iou', stats.meanIoU);
        el('stat-accuracy', stats.pixelAccuracy);
        el('stat-dice', stats.diceScore);
        el('stat-speed', stats.inferenceTimeMs, 'ms');
    }

    document.querySelectorAll('.stat-value').forEach(el => {
        const target = parseFloat(el.dataset.target);
        const suffix = el.dataset.suffix || '';
        const decimals = suffix ? 1 : 4;
        const duration = 2000;
        const startTime = performance.now();
        function tick(now) {
            const progress = Math.min((now - startTime) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            el.textContent = (target * eased).toFixed(decimals) + suffix;
            if (progress < 1) requestAnimationFrame(tick);
        }
        requestAnimationFrame(tick);
    });
    setTimeout(() => {
        document.querySelectorAll('.stat-bar-fill').forEach(el => el.classList.add('animated'));
    }, 300);
}

// ---- Scroll Observer ----

function initScrollObserver() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                entry.target.querySelectorAll('.class-iou-bar-fill').forEach(bar => {
                    setTimeout(() => { bar.style.width = bar.dataset.target + '%'; }, 200);
                });
            }
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });
    document.querySelectorAll('.reveal').forEach(el => observer.observe(el));
}

function initNavScroll() {
    const sections = document.querySelectorAll('.section');
    const navLinks = document.querySelectorAll('.nav-link');
    const navbar = document.getElementById('navbar');
    window.addEventListener('scroll', () => {
        navbar.classList.toggle('scrolled', window.scrollY > 50);
        let current = '';
        sections.forEach(s => { if (window.scrollY >= s.offsetTop - 100) current = s.id; });
        navLinks.forEach(l => l.classList.toggle('active', l.getAttribute('href') === '#' + current));
    });
}

// ---- Model Config Display ----

function updateArchitecture() {
    if (!USE_REAL) return;
    const cfg = REAL_DATA.modelConfig;
    // Update backbone label
    const backboneNode = document.querySelector('.arch-highlight .arch-label');
    if (backboneNode) backboneNode.textContent = cfg.backbone;
    const backboneDetail = document.querySelector('.arch-highlight .arch-detail');
    if (backboneDetail) backboneDetail.textContent = 'Frozen Backbone';
}

// ---- Data Source Banner ----

function addDataBanner() {
    if (!USE_REAL) return;
    const banner = document.createElement('div');
    banner.style.cssText = 'position:fixed;bottom:20px;right:20px;z-index:999;background:rgba(74,222,128,0.15);border:1px solid rgba(74,222,128,0.3);color:#4ade80;padding:8px 16px;border-radius:10px;font-size:0.8rem;font-weight:600;backdrop-filter:blur(10px);display:flex;align-items:center;gap:8px;';
    banner.innerHTML = `<span style="width:8px;height:8px;background:#4ade80;border-radius:50%;display:inline-block;animation:pulse 2s infinite"></span> Live Model — IoU: ${REAL_DATA.heroStats.meanIoU}`;
    document.body.appendChild(banner);

    const style = document.createElement('style');
    style.textContent = '@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}';
    document.head.appendChild(style);
}

// ---- Init ----

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initClassGrid();
    initGallery();
    initComparisonSlider();
    initConfusionMatrix();
    initFailureAnalysis();
    animateStats();
    initScrollObserver();
    initNavScroll();
    updateArchitecture();
    addDataBanner();
});
