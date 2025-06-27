async function fetchImage() {
    const image = document.getElementById('display-image');
    const error = document.getElementById('error');
    const processForm = document.getElementById('process-form');
    const imageFileInput = document.getElementById('image-file');

    try {
        const response = await fetch('/image/fetch-image');
        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            image.src = url;
            image.classList.remove('hidden');
            error.textContent = '';

            // Convert blob to file for processing
            const file = new File([blob], 'captured_image.jpg', { type: 'image/jpeg' });
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            imageFileInput.files = dataTransfer.files;

            processForm.classList.remove('hidden');
        } else {
            error.textContent = 'Failed to capture image';
        }
    } catch (err) {
        error.textContent = 'An error occurred. Please try again.';
    }
}

document.getElementById('process-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const error = document.getElementById('error');

    try {
        const response = await fetch('/image/upload', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        if (response.ok && result.redirect) {
            window.location.href = result.redirect;
        } else {
            error.textContent = result.detail || 'Failed to process image';
        }
    } catch (err) {
        error.textContent = 'An error occurred. Please try again.';
    }
});