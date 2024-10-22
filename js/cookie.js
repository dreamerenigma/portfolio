const cookieBox = document.querySelector(".cookie-popup"),
   buttons = document.querySelectorAll(".lng-button-accept, .lng-button-decline");

const executeCodes = () => {
   // if cookie contains visit it will be returned and below of this code will not run
   if (document.cookie.includes("visit")) return;
   cookieBox.classList.add("show");

   buttons.forEach((button) => {
      button.addEventListener("click", () => {
         cookieBox.classList.remove("show");

         // if button has acceptBtn id
         if (button.id == "acceptBtn") {
            // set cookie for month. 60 = 1 min, 60 = 1 hours, 24 = 1 day, 30 = 30 days
            document.cookie = "cookieBy= visit; max-age=" + 60 * 60 * 24 * 30;
         }
      });
   });
};

// executeCodes function will be called on webpage load
window.addEventListener("load", executeCodes);
