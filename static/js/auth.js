async function getTranslation(key, lang) {
    try {
        const response = await fetch(`/translations?key=${encodeURIComponent(key)}&lang=${lang}`);
        const result = await response.json();
        return result.translation || key;
    } catch {
        return key;
    }
}

async function register() {
    const email = document.getElementById('email').value;
    const name = document.getElementById('name').value;
    const birthday = document.getElementById('birthday').value;
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirm-password').value;
    const error = document.getElementById('error');
    const lang = new URLSearchParams(window.location.search).get('lang') || 'en';

    try {
        const response = await fetch('/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, name, birthday, password, confirm_password: confirmPassword })
        });
        const result = await response.json();
        if (response.ok) {
            document.getElementById('register-form').classList.add('hidden');
            document.getElementById('otp-form').classList.remove('hidden');
            error.textContent = '';
        } else {
            error.textContent = await getTranslation(result.detail, lang);
        }
    } catch (err) {
        error.textContent = await getTranslation('An error occurred. Please try again.', lang);
    }
}

async function verifyOTP() {
    const email = document.getElementById('email').value;
    const code = document.getElementById('otp').value;
    const error = document.getElementById('error');
    const lang = new URLSearchParams(window.location.search).get('lang') || 'en';

    try {
        const response = await fetch('/auth/verify-register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, code })
        });
        const result = await response.json();
        if (response.ok) {
            window.location.href = result.redirect;
        } else {
            error.textContent = await getTranslation(result.detail, lang);
        }
    } catch (err) {
        error.textContent = await getTranslation('An error occurred. Please try again.', lang);
    }
}

async function resendOTP() {
    const email = document.getElementById('email').value;
    const error = document.getElementById('error');
    const lang = new URLSearchParams(window.location.search).get('lang') || 'en';

    try {
        const response = await fetch('/auth/resend-otp', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(email)
        });
        const result = await response.json();
        if (response.ok) {
            error.textContent = await getTranslation('OTP resent successfully', lang);
            error.classList.remove('text-red-500');
            error.classList.add('text-green-500');
        } else {
            error.textContent = await getTranslation(result.detail, lang);
            error.classList.add('text-red-500');
            error.classList.remove('text-green-500');
        }
    } catch (err) {
        error.textContent = await getTranslation('An error occurred. Please try again.', lang);
    }
}

async function login() {
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const error = document.getElementById('error');
    const lang = new URLSearchParams(window.location.search).get('lang') || 'en';

    try {
        const response = await fetch('/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ email, password })
        });
        const result = await response.json();
        if (response.ok) {
            window.location.href = result.redirect;
        } else {
            error.textContent = await getTranslation(result.detail, lang);
        }
    } catch (err) {
        error.textContent = await getTranslation('An error occurred. Please try again.', lang);
    }
}