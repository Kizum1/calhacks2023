// Get the video element by its ID
const videoElement = document.getElementById('videoElement');

// Access the camera stream
navigator.mediaDevices.getUserMedia({ video: true })
  .then(function(stream) {
    // Set the video element's source to the camera stream
    videoElement.srcObject = stream;
  })
  .catch(function(error) {
    console.error('Error accessing camera: ', error);
  });
