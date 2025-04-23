let audioChunks = [];
let mediaRecorder;

document.getElementById('recordBtn').addEventListener('click', () => {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            audioChunks = [];

            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });

            mediaRecorder.addEventListener("stop", () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                reader.onloadend = () => {
                    document.getElementById('audio_file').value = reader.result;
                    document.getElementById('audioPreview').src = reader.result;
                };
            });

            setTimeout(() => mediaRecorder.stop(), 3000); // Record for 3s
        });
});

document.getElementById('captureBtn').addEventListener('click', () => {
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);
    const imageData = canvas.toDataURL('image/png');
    document.getElementById('image_data').value = imageData;
});

navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    const video = document.getElementById('video');
    video.srcObject = stream;
});

function logTask(task) {
    const employee_id = document.querySelector('input[name="employee_id"]').value;
    const emotion = "{{ emotion }}";  // Will be rendered on the page

    fetch('/log_task', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ employee_id, emotion, task })
    });
}
