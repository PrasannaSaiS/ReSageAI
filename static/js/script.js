'use strict';

// ── Upload page: drag-and-drop + file selection ────────────────────────────────
(function () {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('fileInput');
    const fileNameDisplay = document.getElementById('file-name-display');
    const submitBtn = document.getElementById('submitBtn');
    const uploadForm = document.getElementById('uploadForm');

    if (!dropArea) return;

    const ALLOWED_TYPES = [
        'application/pdf',
        'application/msword',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    ];

    const isValidFile = (file) =>
        ALLOWED_TYPES.includes(file.type) || /\.(pdf|doc|docx)$/i.test(file.name);

    const showFileName = () => {
        const file = fileInput.files && fileInput.files[0];
        if (file) {
            fileNameDisplay.textContent = `✓  ${file.name}`;
            fileNameDisplay.style.color = 'var(--success)';
        } else {
            fileNameDisplay.textContent = '';
        }
    };

    const handleFiles = (files) => {
        if (!files || files.length === 0) return;
        const file = files[0];
        if (isValidFile(file)) {
            // DataTransfer trick to assign files to input
            try {
                const dt = new DataTransfer();
                dt.items.add(file);
                fileInput.files = dt.files;
            } catch (_) {
                // DataTransfer not supported — drag-drop won't work but click-select still does
            }
            showFileName();
        } else {
            fileNameDisplay.textContent = '✗  Invalid file type. Please use PDF, DOC, or DOCX.';
            fileNameDisplay.style.color = 'var(--danger)';
        }
    };

    // Drag events
    ['dragenter', 'dragover'].forEach((ev) =>
        dropArea.addEventListener(ev, (e) => {
            e.preventDefault();
            dropArea.classList.add('dragover');
        })
    );

    ['dragleave', 'dragend'].forEach((ev) =>
        dropArea.addEventListener(ev, () => dropArea.classList.remove('dragover'))
    );

    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dropArea.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    if (fileInput) fileInput.addEventListener('change', showFileName);

    // Disable submit while uploading
    if (uploadForm && submitBtn) {
        uploadForm.addEventListener('submit', () => {
            if (!fileInput.files || !fileInput.files[0]) return;
            submitBtn.disabled = true;
            submitBtn.textContent = 'Uploading…';
        });
    }
})();

// ── Analysis page: step animation + redirect to results ───────────────────────
(function () {
    const loaderCard = document.querySelector('[data-filename]');
    if (!loaderCard) return;

    const filename = JSON.parse(loaderCard.getAttribute('data-filename'));
    const resultsUrl = '/results?filename=' + encodeURIComponent(filename);

    const steps = [
        document.getElementById('step1'),
        document.getElementById('step2'),
        document.getElementById('step3'),
        document.getElementById('step4'),
    ];

    // Staggered step animation timings (ms)
    const timings = [0, 600, 1400, 2200];

    steps.forEach(function (el, i) {
        if (!el) return;
        setTimeout(function () {
            if (i > 0 && steps[i - 1]) steps[i - 1].classList.replace('active', 'done');
            el.classList.add('active');
        }, timings[i]);
    });

    // Redirect after steps finish
    setTimeout(function () {
        if (steps[steps.length - 1]) steps[steps.length - 1].classList.replace('active', 'done');
        window.location.href = resultsUrl;
    }, 3000);
})();
