var latitude;
var longitude;
const api_url = window.location.href

function getCoordinates() {
    fetch(`${api_url}coo`)
        .then(response => response.json())
        .then(data => {
            latitude = data.latitude.toFixed(2)
            longitude = data.longitude.toFixed(2)
            document.getElementById('coordinates').textContent = `Широта: ${latitude}, Долгота: ${longitude}`;
        });
}

function userCoordinates() {
    latitude = document.getElementById('latitude').value;
    longitude = document.getElementById('longitude').value;

    if (latitude  != "" && longitude  != "") {
        document.getElementById('coordinates').textContent = `Широта: ${latitude}, Долгота: ${longitude}`;
    } else {
        document.getElementById('coordinates').textContent = 'Пожалуйста, введите координаты.';
    }
}

function getCountryDescription(model) {
    if (latitude  === undefined || longitude  === undefined) {
        document.getElementById('coordinates').innerText = 'Пожалуйста, введите координаты.';
        return;
    }
    const start = new Date().getTime();
    fetch(`${api_url}des?latitude=${latitude}&longitude=${longitude}&model=${model}`)
        .then(response => response.json())
        .then(data => {
            const end = new Date().getTime();
            const response_time = (end - start)/1000;
            document.getElementById(`${model}-description`).textContent = `${data.description}\nВремя ответа: ${response_time}`
        });
}
