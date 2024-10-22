class Chatbox {
  constructor() {
    this.args = {
      openButton: document.querySelector('.chatbox__button'),
      chatBox: document.querySelector('.chatbox__support'),
      sendButton: document.querySelector('.send__button'),
      chatbotCloseBtn: document.querySelector('.close-btn'),
      chatboxSvgBot: document.querySelector('.chatbox-svg-bot'),
      chatboxSvgMessage: document.querySelector('.chatbox-svg-message')
    };

    this.state = false;
    this.messages = [];
  }

  display() {
    const { openButton, chatBox, sendButton, chatbotCloseBtn } = this.args;

    openButton.addEventListener('click', (event) => {
      event.stopPropagation(); // Stop the event propagation
      this.toggleState(chatBox);
      this.toggleIcons();
    });

    sendButton.addEventListener('click', () => this.onSendButton(chatBox));

    chatbotCloseBtn.addEventListener('click', () => this.closeChatbot(chatBox));

    document.body.addEventListener('click', (event) => this.handleOutsideClick(event));

    const node = chatBox.querySelector('input');
    node.addEventListener("keyup", ({ key }) => {
      if (key === "Enter") {
        this.onSendButton(chatBox);
      }
    });
  }

  closeChatbot(chatbox) {
    this.state = !this.state;

    // show or hide the box
    if (this.state) {
      chatbox.classList.add('chatbox--active');
    } else {
      chatbox.classList.remove('chatbox--active');
    }

    this.toggleIcons();
  }

  toggleState(chatbox) {
    this.state = !this.state;

    // show or hide the box
    if (this.state) {
      chatbox.classList.add('chatbox--active');
    } else {
      chatbox.classList.remove('chatbox--active');
    }
  }

  toggleIcons() {
    const { chatboxSvgMessage, chatboxSvgBot } = this.args;

    if (this.state) {
      // Hide chatbox-svg-message and show chatbox-svg-bot
      chatboxSvgMessage.style.display = 'none';
      chatboxSvgBot.style.display = 'block';
    } else {
      // Show chatbox-svg-message and hide chatbox-svg-bot
      chatboxSvgMessage.style.display = 'block';
      chatboxSvgBot.style.display = 'none';
    }

    if (this.state) {
      // If chatbox is active, hide chatbox-svg-message
      chatboxSvgMessage.style.display = 'none';
    }
  }

  onSendButton(chatbox) {
    var textField = chatbox.querySelector('input');
    let text1 = textField.value;
    if (text1 === "") {
      return;
    }

    let msg1 = { name: "User", message: text1 };
    this.messages.push(msg1);

    fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      body: JSON.stringify({ message: text1 }),
      mode: 'cors',
      headers: {
        'Content-Type': 'application/json'
      },
    })
      .then(r => r.json())
      .then(r => {
        let msg2 = { name: "Sam", message: r.answer };
        this.messages.push(msg2);
        this.updateChatText(chatbox);
        textField.value = '';
      })
      .catch((error) => {
        console.error('Error:', error);
        this.updateChatText(chatbox);
        textField.value = '';
      });
  }

  updateChatText(chatbox) {
    var html = '';
    this.messages.slice().reverse().forEach(function (item, index) {
      if (item.name === "Sam") {
        html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>';
      } else {
        html += '<div class="messages__item messages__item--operator">' + item.message + '</div>';
      }
    });

    const chatmessage = chatbox.querySelector('.chatbox__messages');
    chatmessage.innerHTML = html;
  }

  handleOutsideClick(event) {
    const { chatBox } = this.args;

    if (!chatBox.contains(event.target) && this.state) {
      // If clicked outside chatBox and chatbox is active, close chatbox
      this.closeChatbot(chatBox);
    }
  }
}

const chatbox = new Chatbox();
chatbox.display();
