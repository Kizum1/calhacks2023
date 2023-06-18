const dynamicTextElement = document.getElementById('dynamicText');
const changeButton = document.getElementById('changeButton');

const texts = [
    'Hello',
    'Thank You',
    'I love you',
    'Sorry',
    'Goodbye'
];

let currentIndex = 0;

function changeText() {
  dynamicTextElement.textContent = texts[currentIndex];
  currentIndex = (currentIndex + 1) % texts.length;
}

changeButton.addEventListener('click', changeText);