document.addEventListener('DOMContentLoaded', function() {
    console.log('Script loaded successfully');
    
    // Основные элементы интерфейса
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const uploadPlaceholder = document.getElementById('uploadPlaceholder');
    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const removeImage = document.getElementById('removeImage');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultContainer = document.getElementById('resultContainer');
    const resultContent = document.getElementById('resultContent');
    const loader = document.getElementById('loader');
    
    // Элементы языкового переключателя
    const langRuBtn = document.getElementById('langRu');
    const langEnBtn = document.getElementById('langEn');
    
    // Элементы модального окна помощи
    const fabHelp = document.getElementById('fabHelp');
    const modalOverlay = document.getElementById('modalOverlay');
    const modalClose = document.getElementById('modalClose');
    
    console.log('Modal elements check:', { 
        fabHelp: !!fabHelp, 
        modalOverlay: !!modalOverlay, 
        modalClose: !!modalClose 
    });
    
    // Объект переводов
    const translations = {
        en: {
            title: 'Image Analysis System',
            uploadText: 'Click to upload or drag image here',
            analyzeButton: 'Analyze Image',
            resultTitle: 'Analysis Result',
            analyzingText: 'Analyzing image...',
            errorNetwork: 'Failed to analyze image. Please try again.',
            errorFileType: 'Please select an image file.',
            confidenceLabel: 'Confidence:',
            classLabel: 'Class:',
            descriptionLabel: 'Description:',
            teamInfo: '5+5=11 Team, Moscow Aviation Institute (MAI), 2025'
        },
        ru: {
            title: 'Система анализа изображений',
            uploadText: 'Нажмите для загрузки или перетащите изображение сюда',
            analyzeButton: 'Анализировать изображение',
            resultTitle: 'Результат анализа',
            analyzingText: 'Анализ изображения...',
            errorNetwork: 'Не удалось проанализировать изображение. Попробуйте еще раз.',
            errorFileType: 'Пожалуйста, выберите файл изображения.',
            confidenceLabel: 'Уверенность:',
            classLabel: 'Класс:',
            descriptionLabel: 'Описание:',
            teamInfo: 'Команда 5+5=11, Московский авиационный институт (МАИ), 2025'
        }
    };
    
    // Текущий язык и выбранный файл
    let currentLanguage = localStorage.getItem('preferredLanguage') || 'en';
    let selectedFile = null;
    
    console.log('Initial language:', currentLanguage);
    
    // Проверка доступности элементов
    if (!langRuBtn || !langEnBtn) {
        console.error('Language buttons not found in DOM!');
        return;
    }
    
    if (!fabHelp || !modalOverlay || !modalClose) {
        console.error('Modal elements not found in DOM!');
        console.error('Missing elements:', {
            fabHelp: !fabHelp ? 'MISSING' : 'OK',
            modalOverlay: !modalOverlay ? 'MISSING' : 'OK', 
            modalClose: !modalClose ? 'MISSING' : 'OK'
        });
        return;
    }
    
    // Инициализация языка при загрузке страницы
    switchLanguage(currentLanguage);
    
    // *** ОБРАБОТЧИКИ ДЛЯ МОДАЛЬНОГО ОКНА ***
    fabHelp.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        console.log('FAB Help clicked - opening modal');
        openModal();
    });
    
    modalClose.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        console.log('Modal close button clicked');
        closeModal();
    });
    
    // Закрытие модального окна при клике на фон
    modalOverlay.addEventListener('click', function(e) {
        if (e.target === modalOverlay) {
            console.log('Clicked on modal background');
            closeModal();
        }
    });
    
    // Закрытие модального окна при нажатии Escape
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && !modalOverlay.hidden) {
            console.log('Escape pressed - closing modal');
            closeModal();
        }
    });
    
    console.log('Event listeners attached successfully');
    
    // *** ФУНКЦИИ МОДАЛЬНОГО ОКНА *** 
	function openModal() {
		modalOverlay.style.display = 'flex';
		document.body.style.overflow = 'hidden';
	}

	function closeModal() {
		modalOverlay.style.display = 'none';
		document.body.style.overflow = '';
	}

    
    // Обработчики событий для языковых кнопок
    langRuBtn.addEventListener('click', function(e) {
        e.preventDefault();
        console.log('RU button clicked');
        switchLanguage('ru');
    });
    
    langEnBtn.addEventListener('click', function(e) {
        e.preventDefault();
        console.log('EN button clicked');
        switchLanguage('en');
    });
    
    // Функция переключения языка
    function switchLanguage(lang) {
        console.log('Switching to language:', lang);
        currentLanguage = lang;
        
        // Удаляем active класс со всех кнопок
        langRuBtn.classList.remove('active');
        langEnBtn.classList.remove('active');
        
        // Добавляем active класс к нужной кнопке
        if (lang === 'ru') {
            langRuBtn.classList.add('active');
        } else {
            langEnBtn.classList.add('active');
        }
        
        // Обновление всех элементов с data-атрибутами
        const elementsToTranslate = document.querySelectorAll('[data-en][data-ru]');
        console.log('Found elements to translate:', elementsToTranslate.length);
        
        elementsToTranslate.forEach((element, index) => {
            const translation = element.getAttribute('data-' + lang);
            if (translation) {
                const oldText = element.textContent;
                element.textContent = translation;
                console.log(`Element ${index}: "${oldText}" → "${translation}"`);
            }
        });
        
        // Обновление title страницы
        const titleElement = document.querySelector('title[data-en][data-ru]');
        if (titleElement) {
            const titleTranslation = titleElement.getAttribute('data-' + lang);
            if (titleTranslation) {
                document.title = titleTranslation;
                console.log('Page title updated to:', titleTranslation);
            }
        }
        
        // Обновление текста в loader, если он видим
        const loaderText = document.querySelector('#loader p');
        if (loaderText && !loader.hidden) {
            loaderText.textContent = translations[lang].analyzingText;
        }
        
        // Сохранение предпочтений пользователя
        localStorage.setItem('preferredLanguage', lang);
        console.log('Language preferences saved:', lang);
    }
    
    // Обработчик клика на область загрузки
    uploadArea.addEventListener('click', function() {
        if (!previewContainer.hidden) return;
        imageInput.click();
    });
    
    // Обработчики drag and drop
    uploadArea.addEventListener('dragover', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#3498db';
        uploadArea.style.backgroundColor = 'rgba(52, 152, 219, 0.05)';
    });
    
    uploadArea.addEventListener('dragleave', function() {
        uploadArea.style.borderColor = '#ccc';
        uploadArea.style.backgroundColor = 'transparent';
    });
    
    uploadArea.addEventListener('drop', function(e) {
        e.preventDefault();
        uploadArea.style.borderColor = '#ccc';
        uploadArea.style.backgroundColor = 'transparent';
        
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });
    
    // Обработчик выбора файла
    imageInput.addEventListener('change', function() {
        if (this.files.length) {
            handleFile(this.files[0]);
        }
    });
    
    // Обработчик кнопки удаления изображения
    removeImage.addEventListener('click', function(e) {
        e.stopPropagation();
        resetUpload();
    });
    
    // Обработчик кнопки анализа
    analyzeBtn.addEventListener('click', function() {
        if (!selectedFile) return;
        
        // Показ загрузчика с локализованным текстом
        loader.hidden = false;
        const loaderText = document.querySelector('#loader p');
        loaderText.textContent = translations[currentLanguage].analyzingText;
        
        resultContainer.hidden = true;
        analyzeBtn.disabled = true;
        
        // Создание FormData для отправки
        const formData = new FormData();
        formData.append('image', selectedFile);
        
        // Отправка запроса на backend
        fetch('http://localhost:8000/api/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            displayResult(data);
        })
        .catch(error => {
            console.error('Error:', error);
            displayResult({ 
                error: translations[currentLanguage].errorNetwork 
            });
        })
        .finally(() => {
            loader.hidden = true;
            analyzeBtn.disabled = false;
        });
    });
    
    // Функция обработки файла
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert(translations[currentLanguage].errorFileType);
            return;
        }
        
        selectedFile = file;
        
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
            uploadPlaceholder.hidden = true;
            previewContainer.hidden = false;
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }
    
    // Функция сброса загрузки
    function resetUpload() {
        selectedFile = null;
        imageInput.value = '';
        uploadPlaceholder.hidden = false;
        previewContainer.hidden = true;
        analyzeBtn.disabled = true;
        resultContainer.hidden = true;
    }
    
    // Функция отображения результатов анализа
    function displayResult(result) {
        resultContainer.hidden = false;
        
        if (result.error) {
            resultContent.innerHTML = `<p class="error">${result.error}</p>`;
            return;
        }
        
        let html = '';
        
        if (result.class !== undefined) {
            html += `<p><strong>${translations[currentLanguage].classLabel}</strong> ${result.class}</p>`;
        }
        
        if (result.confidence !== undefined) {
            const confidence = (result.confidence * 100).toFixed(2);
            html += `<p><strong>${translations[currentLanguage].confidenceLabel}</strong> ${confidence}%</p>`;
        }
        
        if (result.description) {
            html += `<p><strong>${translations[currentLanguage].descriptionLabel}</strong> ${result.description}</p>`;
        }
        
        resultContent.innerHTML = html || JSON.stringify(result, null, 2);
    }
    
    console.log('Script initialization completed successfully');
});
