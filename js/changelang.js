/*======================== LANGUAGE SWITCHING MODE ========================*/
const langmenu = document.querySelector(".languageSelect");
const allLang = ["ru", "us", "es"];

langmenu.addEventListener("change", changeLanguage);

function changeLanguage() {
  let lang = langmenu.value;
  updateContent(lang);
}

function updateContent(lang) {
  console.log("Selected language:", lang);

  if (!allLang.includes(lang)) {
    lang = defaultLang;
  }

  // Обновляем контент на странице
  for (let key in langArr) {
    if (Object.prototype.hasOwnProperty.call(langArr, key)) {
      let elem = document.querySelector(".lng-" + key);
      if (elem) {
        elem.innerHTML = langArr[key][lang];
        console.log(
          "Updated element:",
          key,
          "with content:",
          langArr[key][lang]
        );
      } else {
      }
    }
  }

  const linkElement = document.querySelector('.lng-view-details-1 a');
  if (linkElement) {
    const linkTranslationKey = "view-details-1";
    const linkTranslation = langArr[linkTranslationKey] && langArr[linkTranslationKey][lang];
    linkElement.textContent = linkTranslation;
  }

  // Обновляем флаг языка
  const selectedLangFlag = document.getElementById('selectedLangFlag');
  if (selectedLangFlag) {
    selectedLangFlag.src = `https://i.ibb.co/${langArr.flagImages[lang]}.png`;
  }

  // Теперь динамически меняем текст внутри элемента с id "selectedLangText"
  const selectedLangText = document.getElementById('selectedLangText');
  if (selectedLangText) {
    selectedLangText.textContent = langArr.selectedLangText[lang];
  }

  const inputElement = document.querySelector('[data-translate="lng-write-message"]');
  if (inputElement) {
    inputElement.placeholder = langArr['write-message'][lang];
  }

  // Обновляем содержимое кнопок, которые имеют атрибут data-translate
  const buttonsWithTranslation = document.querySelectorAll('[data-translate]');
  buttonsWithTranslation.forEach(button => {
    const buttonTextElement = button.querySelector(".lng-details");
    if (buttonTextElement) {
      buttonTextElement.textContent = langArr.details && langArr.details[lang] || "";
    }
  });

  const readMoreLink = document.querySelector('.lng-button-read-more');
  if (readMoreLink) {
    const cookiePolicyURL = `https://inputstudios.ru/cookie_policy-${lang}.html`;

    // Handle click event
    readMoreLink.addEventListener('click', function (event) {
      // Prevent the default behavior (navigation)
      event.preventDefault();

      // Open the cookie policy page based on the selected language
      window.open(cookiePolicyURL, '_blank');
    });
  }

  swiper.destroy(true, true);

  // Reinitialize Swiper with updated content
  swiper = new Swiper(".testimonial__container", {
    loop: true,
    grabCursor: true,
    spaceBetween: 24,
    pagination: {
      el: ".swiper-pagination",
      clickable: true,
      dynamicBullets: true,
    },
    breakpoints: {
      576: {
        slidesPerView: 2,
      },
      768: {
        slidesPerView: 2,
        spaceBetween: 48,
      },
    },
  });
}

// Получаем текущий выбранный язык из localStorage при загрузке страницы
document.addEventListener('DOMContentLoaded', function () {
  const storedLang = localStorage.getItem('selectedLang');
  const defaultLang = 'ru'; // Язык по умолчанию
  const initialLang = storedLang || defaultLang;
  updateContent(initialLang);

  const customSelect = document.querySelector('.custom-select');
  const selectedLangText = document.getElementById('selectedLangText');
  const selectedLangFlag = document.getElementById('selectedLangFlag');
  const options = document.querySelectorAll('.option');

  customSelect.addEventListener('click', function (event) {
    event.stopPropagation();
    customSelect.classList.toggle('open');
  });

  options.forEach(function (option) {
    option.addEventListener('click', function () {
      const value = option.getAttribute('data-value');
      selectedLangText.textContent = option.textContent.trim();
      selectedLangFlag.src = option.querySelector('img').src;
      customSelect.classList.remove('open');

      // Сохраняем выбранный язык в localStorage
      localStorage.setItem('selectedLang', value);

      updateContent(value);
    });
  });

  document.addEventListener('click', function (event) {
    const target = event.target;
    if (!customSelect.contains(target)) {
      customSelect.classList.remove('open');
    }
  });
});
