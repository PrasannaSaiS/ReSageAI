// script.js
const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("fileInput");
const fileNameDisplay = document.getElementById("file-name-display");

dropArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropArea.classList.add("dragover");
});

dropArea.addEventListener("dragleave", () => {
    dropArea.classList.remove("dragover");
});

dropArea.addEventListener("drop", (e) => {
    e.preventDefault();
    dropArea.classList.remove("dragover");
    fileInput.files = e.dataTransfer.files;
    fileNameDisplay.textContent = "Selected: " + fileInput.files[0].name;
});

fileInput.addEventListener("change", () => {
    fileNameDisplay.textContent = "Selected: " + fileInput.files[0].name;
});
