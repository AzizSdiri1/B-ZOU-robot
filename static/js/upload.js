async function getTranslation(key, lang) {
    try {
        const response = await fetch(`/translations?key=${encodeURIComponent(key)}&lang=${lang}`);
        const result = await response.json();
        return result.translation || key;
    } catch {
        return key;
    }
}

document.getElementById('upload-form').addEventListener('submit', async function (e) {
    e.preventDefault();
    
    const file = document.getElementById('file-input').files[0];
    const message = document.getElementById('message');
    const error = document.getElementById('error');
    const lang = new URLSearchParams(window.location.search).get('lang') || 'en';

    if (!file) {
        error.textContent = await getTranslation('Please select an image to upload.', lang);
        message.textContent = '';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/image/upload', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        if (response.ok) {
            message.textContent = await getTranslation('Image uploaded successfully! Redirecting to result...', lang);
            error.textContent = '';
            setTimeout(() => {
                window.location.href = `/result/${result.filename}?lang=${lang}`;
            }, 1000);
        } else {
            error.textContent = await getTranslation(result.detail, lang);
            message.textContent = '';
        }
    } catch (err) {
        error.textContent = await getTranslation('An error occurred. Please try again.', lang);
        message.textContent = '';
    }
});