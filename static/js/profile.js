async function getTranslation(key, lang) {
    try {
        const response = await fetch(`/translations?key=${encodeURIComponent(key)}&lang=${lang}`);
        const result = await response.json();
        return result.translation || key;
    } catch {
        return key;
    }
}

async function updateProfile() {
    const name = document.getElementById('name').value;
    const birthday = document.getElementById('birthday').value;
    const file = document.getElementById('file-input').files[0];
    const message = document.getElementById('message');
    const error = document.getElementById('error');
    const lang = new URLSearchParams(window.location.search).get('lang') || 'en';

    const formData = new FormData();
    formData.append('name', name);
    formData.append('birthday', birthday);
    if (file) {
        formData.append('file', file);
    }

    try {
        const response = await fetch('/auth/update-profile', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        if (response.ok) {
            message.textContent = await getTranslation(result.message, lang);
            error.textContent = '';
            if (file) {
                const reader = new FileReader();
                reader.onload = () => {
                    document.getElementById('profile-image').src = reader.result;
                };
                reader.readAsDataURL(file);
            }
        } else {
            error.textContent = await getTranslation(result.detail, lang);
            message.textContent = '';
        }
    } catch (err) {
        error.textContent = await getTranslation('An error occurred. Please try again.', lang);
        message.textContent = '';
    }
}