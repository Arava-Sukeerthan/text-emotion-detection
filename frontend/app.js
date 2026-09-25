const form = document.getElementById("emotion-form");
const textInput = document.getElementById("text");
const resultBox = document.getElementById("result");

const backendUrl = "https://text-emotion-detection-1-umj6.onrender.com";

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const text = textInput.value.trim();
  if (!text) {
    resultBox.textContent = "Please enter some text first.";
    resultBox.classList.remove("hidden");
    return;
  }

  resultBox.classList.remove("hidden");
  resultBox.innerHTML = "<p>Analyzing...</p>";

  try {
    const response = await fetch(`${backendUrl}/api/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed");
    }

    resultBox.innerHTML = `
      <h2>Emotion: ${data.emotion}</h2>
      <p>Confidence: ${data.confidence}%</p>
    `;
  } catch (error) {
    resultBox.innerHTML = `<p>Error: ${error.message}</p>`;
  }
});
