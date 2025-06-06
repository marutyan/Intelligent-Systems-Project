from pathlib import Path
import os

# ── プロジェクト共通定数 ────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
APP_NAME   = 'no06'          # ★ ここが no06
MEDIA_URL  = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# ── 基本設定 ──────────────────────────────────────
SECRET_KEY = 'django-insecure-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
DEBUG      = True
ALLOWED_HOSTS: list[str] = []

# ── アプリケーション ───────────────────────────────
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    APP_NAME,      # ＝ 'no06'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF    = 'ipro.urls'
WSGI_APPLICATION= 'ipro.wsgi.application'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],      # 追加テンプレートディレクトリが要るならここに
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME':   BASE_DIR / 'db.sqlite3',
    }
}

LANGUAGE_CODE = 'ja'
TIME_ZONE     = 'Asia/Tokyo'
USE_I18N      = True
USE_TZ        = True

STATIC_URL    = '/static/'
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
