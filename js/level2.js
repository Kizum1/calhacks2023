const dynamicTextElement = document.getElementById('dynamicText');
const changeButton = document.getElementById('changeButton');

const texts = [
    '1',
    '2',
    '3',
    '4',
    '5'
];

let currentIndex = 0;

function changeText() {
  dynamicTextElement.textContent = texts[currentIndex];
  currentIndex = (currentIndex + 1) % texts.length;
}

changeButton.addEventListener('click', changeText);