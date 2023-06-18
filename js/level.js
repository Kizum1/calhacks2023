const dynamicTextElement = document.getElementById('dynamicText');
const changeButton = document.getElementById('changeButton');

const texts = [
    'A',
    'B',
    'C',
    'D',
    'E'
];

let currentIndex = 0;

  if (currentWordIndex === words.length - 1) {
    document.getElementById('changeButton').disabled = true;
    
  }

  currentWordIndex++;
function changeText() {
  dynamicTextElement.textContent = texts[currentIndex];
  currentIndex = (currentIndex + 1) % texts.length;
}

changeButton.addEventListener('click', changeText);