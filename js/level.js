const dynamicTextElement = document.getElementById('dynamicText');
const changeButton = document.getElementById('changeButton');

const texts = [
    'Hello, how are you?',
    'Do you want some water?',
    'I like bread',
];

let currentIndex = 0;

function changeText() {
  dynamicTextElement.textContent = texts[currentIndex];
  currentIndex = (currentIndex + 1) % texts.length;
}

changeButton.addEventListener('click', changeText);