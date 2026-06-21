const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('fileInput');
const fileNameDisplay = document.getElementById('file-name-display');

const showFileName = () => {
    const file = fileInput.files && fileInput.files[0];
    fileNameDisplay.textContent = file ? `Selected: ${file.name}` : '';
};

const handleFiles = (files) => {
    if (!files || files.length === 0) {
        return;
    }

    const file = files[0];
    if (file && ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'].includes(file.type) || /\.(pdf|doc|docx)$/i.test(file.name)) {
        fileInput.files = files;
        showFileName();
    } else {
        fileNameDisplay.textContent = 'Invalid file type. Please select a PDF, DOC, or DOCX file.';
    }
};

if (dropArea) {
    dropArea.addEventListener('dragover', (event) => {
        event.preventDefault();
        dropArea.classList.add('dragover');
    });

    dropArea.addEventListener('dragleave', () => {
        dropArea.classList.remove('dragover');
    });

    dropArea.addEventListener('drop', (event) => {
        event.preventDefault();
        dropArea.classList.remove('dragover');
        handleFiles(event.dataTransfer.files);
    });
}

if (fileInput) {
    fileInput.addEventListener('change', showFileName);
}

// Analysis page — redirect to results after step animation
(function () {
    var loaderCard = document.querySelector('[data-filename]');
    if (!loaderCard) return;

    var filename = JSON.parse(loaderCard.getAttribute('data-filename'));
    var resultsUrl = '/results?filename=' + encodeURIComponent(filename);

    var steps = [
        document.getElementById('step1'),
        document.getElementById('step2'),
        document.getElementById('step3'),
        document.getElementById('step4'),
    ];
    var timings = [0, 800, 1800, 2800];

    steps.forEach(function (el, i) {
        if (!el) return;
        setTimeout(function () {
            if (i > 0 && steps[i - 1]) steps[i - 1].classList.replace('active', 'done');
            el.classList.add('active');
        }, timings[i]);
    });

    setTimeout(function () {
        window.location.href = resultsUrl;
    }, 350);
})();
