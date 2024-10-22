const langArr = {
  flagImages: {
    ru: "489wXn1/RU",
    us: "LtPNm0n/US",
    es: "x59Tvcs/ES",
  },
  selectedLangText: {
    ru: "Русский",
    us: "English",
    es: "Inglesa",
  },
  language_ru: {
    ru: "Русский",
    us: "Russian",
    es: "Ruso",
  },
  language_en: {
    ru: "Английский",
    us: "English",
    es: "Inglesa",
  },
  language_es: {
    ru: "Испанский",
    us: "Spanish",
    es: "Español",
  },
  title: {
    ru: "Адаптивное портфолио Input Studios",
    us: "Responsive Portfolio by Input Studios",
    es: "Portafolio receptivo de Input Studios",
  },
  loading: {
    ru: "Загрузка",
    us: "Loading",
    es: "Cargando",
  },
  nav__logo: {
    ru: "Андрей",
    us: "Andrey",
    es: "Andrei",
  },
  nav__home: {
    ru: "Главная",
    us: "Home",
    es: "Hogar",
  },
  nav__about: {
    ru: "О Нас",
    us: "About Us",
    es: "Sobre nosotros",
  },
  nav__skills: {
    ru: "Навыки",
    us: "Skills",
    es: "Habilidades",
  },
  nav__services: {
    ru: "Услуги",
    us: "Services",
    es: "Servicios",
  },
  nav__portfolio: {
    ru: "Портфолио",
    us: "Portfolio",
    es: "Portafolio",
  },
  nav__contactme: {
    ru: "Связь со мной",
    us: "Contact Me",
    es: "Contáctame",
  },
  home__title: {
    ru: "Привет, я Андрей",
    us: "Hi, I'am Andrey",
    es: "Hola, soy Andrey",
  },
  home__subtitle: {
    ru: "Fullstack-разработчик",
    us: "Fullstack-developer",
    es: "Desarrollador de pila completo",
  },
  home__description: {
    ru: "Высокий уровень знаний в области веб-дизайна и разработки, качество выполненных заказов.",
    us: "High level of knowledge in the field of web design and development, quality of completed orders.",
    es: "Alto nivel de conocimiento en el campo del diseño y desarrollo web, calidad de los pedidos completados.",
  },
  home__send: {
    ru: "Связь со мной",
    us: "Contact Me",
    es: "Contáctame",
  },
  "home__scroll-name": {
    ru: "Прокрутить вниз",
    us: "Scroll down",
    es: "Desplácese hacia abajo",
  },
  section__about: {
    ru: "Обо мне",
    us: "About Me",
    es: "Acerca de mí",
  },
  section__introduction: {
    ru: "Мое введение",
    us: "My introduction",
    es: "Mi introducción",
  },
  about__description: {
    ru: "Веб-разработчик, обладающий обширными знаниями и многолетним опытом работы в области веб-технологий и UX/UI-дизайна, выполняющий качественно заказы.",
    us: "Web developer, with extensive knowledge and many years of experience in the field of web technologies and UX/UI design, who fulfills orders with high quality.",
    es: "Un desarrollador web con amplios conocimientos y muchos años de experiencia en el campo de las tecnologías web y el diseño UX/UI, que cumple con los pedidos con alta calidad.",
  },
  "about__info-year": {
    ru: "Многолетний <br> опыт",
    us: "Years <br> experience",
    es: "Años de <br> experiencia",
  },
  "about__info-completed": {
    ru: "Завершенных <br> проектов",
    us: "Completed <br> projects",
    es: "Proyectos <br> completados",
  },
  "about__info-companies": {
    ru: "Компании <br> работали",
    us: "Companies <br> worked",
    es: "Empresas <br> trabajadas",
  },
  download__cv: {
    ru: "Скачать резюме",
    us: "Download CV",
    es: "Descargar currículum",
  },
  section__skills: {
    ru: "Навыки",
    us: "Skills",
    es: "Habilidades",
  },
  section__level: {
    ru: "Мой технический уровень",
    us: "My techical level",
    es: "Mi nivel tecnico",
  },
  skills__frontend: {
    ru: "Фронтенд-разработчик",
    us: "Frontend developer",
    es: "Desarrolladora frontend",
  },
  skills__more__front: {
    ru: "Более 4 лет",
    us: "More than 4 years",
    es: "Más de 4 años",
  },
  skills__backend: {
    ru: "Бэкенд-разработчик",
    us: "Backend developer",
    es: "Desarrollador back-end",
  },
  skills__more__back: {
    ru: "Более 2 лет",
    us: "More than 2 years",
    es: "Mas de 2 años",
  },
  skills__designer: {
    ru: "Дизайнер",
    us: "Designer",
    es: "Diseñador",
  },
  skills__more__design: {
    ru: "Более 5 лет",
    us: "More than 5 years",
    es: "Mas de 5 años",
  },
  section__qualification: {
    ru: "Квалификация",
    us: "Qualification",
    es: "Calificación",
  },
  section__journey: {
    ru: "Мое личное путешествие",
    us: "My personal journey",
    es: "Mi viaje personal",
  },
  education: {
    ru: "Образование",
    us: "Education",
    es: "Educación",
  },
  work: {
    ru: "Работа",
    us: "Work",
    es: "Trabajar",
  },
  qualification__computer: {
    ru: "Компьютерный инженер",
    us: "Computer Enginner",
    es: "Ingeniero informático",
  },
  qualification__colledge: {
    ru: "Технический колледж - Ульяновск",
    us: "Techical Colledge - Ulyanovsk",
    es: "Colegio Técnico - Ulyanovsk",
  },
  qualification__design: {
    ru: "Веб-дизайн",
    us: "Web Design",
    es: "Diseño web",
  },
  qualification__skillbox: {
    ru: "Скиллбокс - Москва",
    us: "Skillbox - Moscow",
    es: "Skillbox - Moscú",
  },
  qualification__development: {
    ru: "Веб-разработка",
    us: "Web Development",
    es: "Desarrollo web",
  },
  qualification__insitute: {
    ru: "Технический институт - Ульяновск",
    us: "Techical insitute - Ulyanovsk",
    es: "Instituto técnico - Ulyanovsk",
  },
  qualification__master: {
    ru: "Мастер в UX/UI",
    us: "Master in UX/UI",
    es: "Máster en UX/UI",
  },
  qualification__netology: {
    ru: "Нетология - Ульяновск",
    us: "Netology - Ulyanovsk",
    es: "Netología - Ulyanovsk",
  },
  qualification__software: {
    ru: "Инженер-программист",
    us: "Software Enginner",
    es: "Ingeniero de software",
  },
  qualification__simm: {
    ru: "Симбирск Мплюс - Ульяновск",
    us: "SimbirskMplus - Ulyanovsk",
    es: "SimbirskMplus - Uliánovsk",
  },
  qualification__front: {
    ru: "Фронтенд-разработчик",
    us: "Frontend Developer",
    es: "Desarrollador frontend",
  },
  qualification__inputstudios: {
    ru: "Input Studios - Ульяновск",
    us: "Input Studios - Ulyanovsk",
    es: "Estudios de entrada - Ulyanovsk",
  },
  qualification__designer: {
    ru: "Дизайнер пользовательского интерфейса",
    us: "UI Designer",
    es: "Diseñador de interfaz de usuario",
  },
  qualification__figma: {
    ru: "Figma - Россия",
    us: "Figma - Russia",
    es: "Figma - Rusia",
  },
  section__services: {
    ru: "Услуги",
    us: "Services",
    es: "Servicios",
  },
  section__offer: {
    ru: "Что я предлагаю",
    us: "What i offer",
    es: "lo que ofrezco",
  },
  services__designer: {
    ru: "UX/UI Дизайн",
    us: "UX/UI Design",
    es: "UX/UI Diseña",
  },
  view__more__designer: {
    ru: "Посмотреть больше",
    us: "View More",
    es: "Ver más",
  },
  "services__modal-designer": {
    ru: "UX/UI Дизайн",
    us: "UX/UI Design",
    es: "UX/UI Diseña",
  },
  service__interface: {
    ru: "Разрабатываю UI дизайн",
    us: "I develop the UI design",
    es: "Desarrollo el diseño UI",
  },
  service__develop: {
    ru: "Разработка сайтов",
    us: "Website development",
    es: "Desarrollo de sitios web",
  },
  service__interactions: {
    ru: "Проектирую UX-взаимодействия",
    us: "I design UX interactions",
    es: "Diseño interacciones UX",
  },
  service__brand: {
    ru: "Разработка прототипов",
    us: "Prototype development",
    es: "Desarrollo de prototipos",
  },
  services__front: {
    ru: "Фронтенд Разработка",
    us: "Frontend Development",
    es: "Desarrollador frontend",
  },
  view__more__front: {
    ru: "Посмотреть больше",
    us: "View More",
    es: "Ver más",
  },
  "services__modal-front": {
    ru: "Фронтенд Разработчик",
    us: "Frontend Developer",
    es: "Desarrollo frontend",
  },
  service__adaptability: {
    ru: "Адаптивность и мобильный дизайн",
    us: "Adaptability and mobile design",
    es: "Adaptabilidad y diseño móvil",
  },
  service__builder: {
    ru: "Сборщик проектов",
    us: "Project Builder",
    es: "Creador de proyectos",
  },
  service__crossbrowser: {
    ru: "Кроссбраузерность",
    us: "Crossbrowser compatibility",
    es: "Compatibilidad entre navegadores",
  },
  service__testing: {
    ru: "Тестирование и отладка",
    us: "Testing and Debugging",
    es: "Pruebas y depuración",
  },
  services__brand: {
    ru: "Брендинг Дизайн",
    us: "Branding Design",
    es: "Diseño de marca",
  },
  view__more__brand: {
    ru: "Посмотреть больше",
    us: "View More",
    es: "Ver más",
  },
  "services__modal-brand": {
    ru: "Брендинг Дизайн",
    us: "Branding Design",
    es: "Diseño de marca",
  },
  service__logo: {
    ru: "Разработка логотипов и иконок",
    us: "Development of logos and icons",
    es: "Desarrollo de logotipos e iconos",
  },
  service__identity: {
    ru: "Разработка айдентики",
    us: "Identity development",
    es: "Desarrollo de identidad",
  },
  service__guideline: {
    ru: "Разработка гайдлайна",
    us: "Guideline development",
    es: "Desarrollo de la guía",
  },
  service__analysis: {
    ru: "Анализ  выполненной работы",
    us: "Analysis of the work performed",
    es: "Análisis del trabajo realizado",
  },
  section__portfolio: {
    ru: "Мое Портфолио",
    us: "My Portfolio",
    es: "Mi portafolio",
  },
  "section__recent-work": {
    ru: "Последние работы",
    us: "Most recent works",
    es: "Trabajos mas recientes",
  },
  portfolio__all: {
    ru: "Все",
    us: "All",
    es: "Todo",
  },
  portfolio__web: {
    ru: "Веб",
    us: "Web",
    es: "Web",
  },
  portfolio__app: {
    ru: "Приложение",
    us: "App",
    es: "Aplicación",
  },
  portfolio__design: {
    ru: "Дизайн",
    us: "Design",
    es: "Diseño",
  },
  portfolio__web: {
    ru: "Веб",
    us: "Web",
    es: "Diseño",
  },
  "portfolio__web-card": {
    ru: "Веб дизайн",
    us: "Web design",
    es: "Diseño web",
  },
  "details-1": {
    ru: "Подробности",
    us: "Details",
    es: "Detalles",
  },
  "portfolio__popup-subtitle-1": {
    ru: "Рекомендуемое — Дизайн",
    us: "Featured — Design",
    es: "Destacado — Diseño",
  },
  "details__title-1": {
    ru: "Дизайн сайта онлайн кинотеатра КиноДом Онлайн",
    us: "Design show online film theater Film Online",
    es: "Diseño de sitio web para el cine online KinoDom Online",
  },
  "details__description-1": {
    ru: "КиноДом Онлайн — это уникальное пространство для ценителей кинематографа, где вы можете наслаждаться широким выбором фильмов и сериалов прямо из уюта своего дома.",
    us: "KinoDom Online is a unique space for cinema lovers, where you can enjoy a wide selection of films and TV series right from the comfort of your home.",
    es: "KinoDom Online es un espacio único para los amantes del cine, donde podrás disfrutar de una amplia selección de películas y series de TV desde la comodidad de tu hogar.",
  },
  "created-details-1": {
    ru: "Создано — 20 авг. 2022 г.",
    us: "Created — 20 aug 2022",
    es: "Creado — 20 de agosto de 2022",
  },
  "techologies-details-1": {
    ru: "Технологии — HTML, CSS, PHP",
    us: "Techologies — HTML, CSS, PHP",
    es: "Tecnología — HTML, CSS, PHP",
  },
  "role-details-1": {
    ru: "Роль — Fullstack-разработчик",
    us: "Role — Fullstack developer",
    es: "Rol — Desarrollador Fullstack",
  },
  "view-details-1": {
    ru: "Посмотреть — ",
    us: "View — ",
    es: "Vista — ",
  },
  "portfolio__app-card-2": {
    ru: "Дизайн приложения",
    us: "App Design",
    es: "Diseño de aplicaciones",
  },
  "details-2": {
    ru: "Подробности",
    us: "Details",
    es: "Detalles",
  },
  "portfolio__popup-subtitle-2": {
    ru: "Рекомендуемое — Дизайн",
    us: "Featured — Design",
    es: "Destacado — Diseño",
  },
  "details__title-2": {
    ru: "Приложение для бесплатного общения Диалог Мессенджер",
    us: "Application for free communication Dialogue Messenger",
    es: "Solicitud de comunicación gratuita Dialogue Messenger",
  },
  "details__description-2": {
    ru: "Приложение для общения с полностью бесплатными функциями для обмена сообщениями, видеочаты, голосовые вызовы, групповые сообщения, редактирование фотографии профиля, статус в профиле.",
    us: "Chatting app with completely free features for messaging, video chats, voice calls, group messages, profile photo editing, profile status.",
    es: "Aplicación de chat con funciones completamente gratuitas para mensajería, videochats, llamadas de voz, mensajes grupales, edición de fotos de perfil, estado de perfil.",
  },
  "created-details-2": {
    ru: "Создано — В работе",
    us: "Created — Work in Progress",
    es: "Creado — Trabajo en progreso",
  },
  "techologies-details-2": {
    ru: "Технологии — Котлин",
    us: "Techologies — Kotlin",
    es: "Tecnologías — Kotlin",
  },
  "role-details-2": {
    ru: "Роль — Fullstack-разработчик",
    us: "Role — Fullstack developer",
    es: "Rol — Desarrollador Fullstack",
  },
  "portfolio__design-card": {
    ru: "Фирменный дизайн",
    us: "Brand Design",
    es: "Diseño de la marca",
  },
  "details-3": {
    ru: "Подробности",
    us: "Details",
    es: "Detalles",
  },
  "portfolio__popup-subtitle-3": {
    ru: "Рекомендуемое — Дизайн",
    us: "Featured — Design",
    es: "Destacado — Diseño",
  },
  "details__title-3": {
    ru: "Приложение для доставки еды не выходя из дома",
    us: "Application for food delivery without leaving home",
    es: "Solicitud de entrega de comida sin salir de casa",
  },
  "details__description-3": {
    ru: "Приложение Food Delivery предоставляет удобный способ заказа еды онлайн и ее доставки к вашему дому. С этим приложением пользователи могут легко выбирать блюда из ресторанов, кафе и закусочных, оформлять заказы и наслаждаться свежей и вкусной едой не выходя из дома.",
    us: "The Food Delivery app provides a convenient way to order food online and have it delivered to your home. With this application, users can easily select dishes from restaurants, cafes and eateries, place orders and enjoy fresh and delicious food without leaving home.",
    es: "La aplicación Food Delivery proporciona una forma conveniente de pedir comida en línea y recibirla en su hogar. Con esta aplicación, los usuarios pueden seleccionar fácilmente platos de restaurantes, cafeterías y comedores, realizar pedidos y disfrutar de comida fresca y deliciosa sin salir de casa.",
  },
  "created-details-3": {
    ru: "Создано — В работе",
    us: "Created — Work in Progress",
    es: "Creado — Trabajo en progreso",
  },
  "techologies-details-3": {
    ru: "Технологии — Dart",
    us: "Techologies — Dart",
    es: "Tecnologías — Dart",
  },
  "role-details-3": {
    ru: "Роль — Fullstack-разработчик",
    us: "Role — Fullstack developer",
    es: "Rol — Desarrollador Fullstack",
  },
  "portfolio__app-card-4": {
    ru: "Дизайн приложения",
    us: "App Design",
    es: "Diseño de aplicaciones",
  },
  "details-4": {
    ru: "Подробности",
    us: "Details",
    es: "Detalles",
  },
  "portfolio__popup-subtitle-4": {
    ru: "Рекомендуемое — Дизайн",
    us: "Featured — Design",
    es: "Destacado — Diseño",
  },
  "details__title-4": {
    ru: "Музыкальный плеер для вопроизведения любых форматов аудио с эквалайзером",
    us: "Music player for playing any audio formats with equalizer",
    es: "Reproductor de música para reproducir cualquier formato de audio con ecualizador",
  },
  "details__description-4": {
    ru: "Maestro — это инновационное музыкальное приложение для платформы Android, созданное для истинных ценителей музыки. Приложение предоставляет уникальный опыт в управлении и прослушивании музыки, объединяя в себе элегантный интерфейс и передовые технологии.",
    us: "Maestro is an innovative music application for the Android platform, created for true music lovers. The app provides a unique experience in managing and listening to music, combining an elegant interface and advanced technology.",
    es: "Maestro es una innovadora aplicación de música para la plataforma Android, creada para verdaderos amantes de la música. La aplicación proporciona una experiencia única en la gestión y escucha de música, combinando una interfaz elegante y tecnología avanzada.",
  },
  "created-details-4": {
    ru: "Создано — 14 июля 2023 г.",
    us: "Created — July 14, 2023",
    es: "Creado — 14 de julio de 2023",
  },
  "techologies-details-4": {
    ru: "Технологии — Котлин",
    us: "Techologies — Kotlin",
    es: "Tecnologías — Kotlin",
  },
  "role-details-4": {
    ru: "Роль — Fullstack-разработчик",
    us: "Role — Fullstack developer",
    es: "Rol — Desarrollador Fullstack",
  },
  project__title: {
    ru: "У вас новый проект",
    us: "You have a new project",
    es: "Tienes un nuevo proyecto",
  },
  project__description: {
    ru: "Свяжитесь со мной сейчас и получите скидку 30% на новый проект.",
    us: "Contact me now and get a 30% discount on your new project.",
    es: "Contáctame ahora y obtén un 30% de descuento en tu nuevo proyecto.",
  },
  project__contactme: {
    ru: "Связь со мной",
    us: "Contact Me",
    es: "Contáctame",
  },
  section__testimonials: {
    ru: "Отзывы",
    us: "Testimonials",
    es: "Testimonials",
  },
  section__saying: {
    ru: "Мои клиенты говорят",
    us: "My clients say",
    es: "Mis clientes dicen",
  },
  "testimonial__name-1": {
    ru: "София Морозова",
    us: "Sofia Morozova",
    es: "Sofía Morozova",
  },
  "testimonial__client-1": {
    ru: "Клиент",
    us: "Client",
    es: "Cliente",
  },
  "testimonial__description-1": {
    ru: "Нужно было сделать дизайн сайта в этом мне помог Андрей Трепалин, в течение суток проблема была решена.",
    us: "It was necessary to make a website design Andrey Trepalin helped me in this within 24 hours the problem was solved.",
    es: "Era necesario hacer un diseño de sitio web Andrey Trepalin me ayudó en esto dentro de las 24 horas se resolvió el problema.",
  },
  "testimonial__name-2": {
    ru: "Мария Сидорова",
    us: "Maria Sidorova",
    es: "María Sidorova",
  },
  "testimonial__client-2": {
    ru: "Клиент",
    us: "Client",
    es: "Cliente",
  },
  "testimonial__description-2": {
    ru: "Выражаю благодарность компании Input Studios за проделанную работу с моим сайтом-портфолио.",
    us: "I express my gratitude to the company Input Studios for the work done with my portfolio site.",
    es: "Expreso mi agradecimiento a la empresa. Input Studios por el trabajo realizado con mi sitio de cartera.",
  },
  "testimonial__name-t3": {
    ru: "Егор Данилов",
    us: "Egor Danilov",
    es: "Egor Danílov",
  },
  "testimonial__client-3": {
    ru: "Клиент",
    us: "Client",
    es: "Cliente",
  },
  "testimonial__description-3": {
    ru: "Заказчик попросил меня сделать новые эффекты для сайта, обратился в Input Studios, мне помог Андрей, спасибо ему за помощь.",
    us: "The customer asked me to make new effects for the site, he turned to Input Studios, Andrey helped me, thanks to him for his help.",
    es: "El cliente me pidió que hiciera nuevos efectos para el sitio, recurrió a Input Studios, Andrey me ayudó, gracias a él por su ayuda.",
  },
  section__contactme: {
    ru: "Свяжитесь со мной",
    us: "Contact Me",
    es: "Contáctame",
  },
  contactme__desc: {
    ru: "Если у вас есть какие-либо вопросы или предложения по сотрудничеству, пожалуйста, заполните форму ниже.",
    us: "If you have any questions or suggestions for cooperation, please fill out the form below.",
    es: "Si tiene alguna pregunta o sugerencia de cooperación, complete el formulario a continuación.",
  },
  whatsapp__callme: {
    ru: "Позвоните мне",
    us: "Call me",
    es: "llámame",
  },
  email__writeme: {
    ru: "Напишите мне",
    us: "Write me",
    es: "Escríbeme",
  },
  vk__writeme: {
    ru: "Напишите мне",
    us: "Write me",
    es: "Escríbeme",
  },
  username: {
    ru: "Имя пользователя",
    us: "Username",
    es: "Nombre de usuario",
  },
  email: {
    ru: "Эл. почта",
    us: "Email",
    es: "Correo electrónico",
  },
  phone: {
    ru: "Телефон",
    us: "Phone",
    es: "Teléfono",
  },
  message: {
    ru: "Сообщение",
    us: "Message",
    es: "Mensaje",
  },
  "send-message": {
    ru: "Отправить сообщение",
    us: "Send Message",
    es: "Enviar mensaje",
  },
  footer__title: {
    ru: "Андрей",
    us: "Andrey",
    es: "Andrei",
  },
  footer__subtitle: {
    ru: "Fullstack разработчик",
    us: "Fullstack developer",
    es: "Desarrollador de pila completa",
  },
  footer__services: {
    ru: "Услуги",
    us: "Services",
    es: "Servicios",
  },
  footer__work: {
    ru: "Работа",
    us: "Work",
    es: "Trabajar",
  },
  footer__contact: {
    ru: "Контакт",
    us: "Contact",
    es: "Contacto",
  },
  privacy: {
    ru: "Политика конфиденциальности",
    us: "Privacy policy",
    es: "Política de privacidad",
  },
  recaptcha: {
    ru: "Этот сайт защищен reCAPTCHA от Яндекс,",
    us: "This site is protected by reCAPTCHA from Yandex,",
    es: "Este sitio está protegido por reCAPTCHA de Yandex,",
  },
  term: {
    ru: "&nbsp;и Условия использования",
    us: "&nbsp;and Terms of Service apply",
    es: "&nbsp;y se aplican los Términos de servicio",
  },
  "cookie-policy": {
    ru: "Политика в отношении файлов cookie",
    us: "Cookie policy",
    es: "Política de cookies",
  },
  footer__right: {
    ru: "Все права защищены",
    us: "All right reserved",
    es: "Todos los derechos reservados",
  },
  "cookie-consent": {
    ru: "Согласие на cookie",
    us: "Cookie Consent",
    es: "Consentimiento de cookies",
  },
  "cookie-description": {
    ru: "Этот веб-сайт использует файлы cookie, чтобы помочь вам лучше и удобнее просматривать веб-сайт.",
    us: "This website use cookie to help you have a superior and more relevant browsing experience on the website.",
    es: "Este sitio web utiliza cookies para ayudarlo a tener una experiencia de navegación superior y más relevante en el sitio web.",
  },
  "button-read-more": {
    ru: "Читать далее...",
    us: "Read more...",
    es: "Leer más...",
  },
  "button-accept": {
    ru: "Принять",
    us: "Accept",
    es: "Aceptar",
  },
  "button-decline": {
    ru: "Отклонить",
    us: "Decline",
    es: "Rechazar",
  },
  "chatbox__heading--header": {
    ru: "Поддержка в чате",
    us: "Chat support",
    es: "Chat de soporte",
  },
  "chatbox__description--header": {
    ru: "Привет. Меня зовут Сэм. Могу я чем-нибудь помочь?",
    us: "Hi. My name is Sam. How can I help you?",
    es: "Hola. Me llamo Sam. ¿Le puedo ayudar en algo?",
  },
  "write-message": {
    ru: "Напишите сообщение...",
    us: "Write a message...",
    es: "Escribe un mensaje...",
  },
  "chatbox__send--footer": {
    ru: "Отправить",
    us: "Send",
    es: "Enviar",
  },
};
