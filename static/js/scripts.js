document.addEventListener("DOMContentLoaded", function() {
    console.log("scripts.js loaded successfully!");

    const modal = document.getElementById("resultModal");
    const span = document.getElementsByClassName("close")[0];

    span.onclick = function() {
        modal.style.display = "none";
    };

    window.onclick = function(event) {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    };

    // Form submission event listener
    document.getElementById('predictionForm').addEventListener('submit', function(event) {
        event.preventDefault();

        const formData = {
            make: document.getElementById('make').value,
            model: document.getElementById('model').value,
            year: document.getElementById('year').value,
            mileage: document.getElementById('mileage').value,
            condition: document.getElementById('condition').value
        };

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        })
        .then(response => response.json())
        .then(data => {
            const modalResult = document.getElementById('modalResult');
            if (data.predicted_price !== undefined && !isNaN(data.predicted_price)) {
                const rupeePrice = data.predicted_price * 74.5; // Assuming conversion rate, adjust as needed
                modalResult.innerText = `Predicted Price: ₹${rupeePrice.toFixed(2)}`; // Displaying in Rupees
            } else if (data.error) {
                modalResult.innerText = `Error: ${data.error}`;
            } else {
                modalResult.innerText = 'An error occurred while predicting the price.';
            }
            modal.style.display = 'block';
            // Clear form inputs after prediction
            document.getElementById('predictionForm').reset();
        })
        .catch(error => {
            console.error('Error:', error);
            const modalResult = document.getElementById('modalResult');
            modalResult.innerText = 'An error occurred while predicting the price.';
            modal.style.display = 'block';
        });
    });
});
