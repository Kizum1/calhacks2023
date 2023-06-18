// JavaScript code for level.js

const words = ['Hello', 'World', 'OpenAI', 'GPT', 'Code'];
let currentWordIndex = 0;

function displayText() {
  const textContainer = document.getElementById('dynamicText');
  textContainer.textContent = words[currentWordIndex];

  if (currentWordIndex === words.length - 1) {
    document.getElementById('changeButton').disabled = true;
  }

  currentWordIndex++;
}
