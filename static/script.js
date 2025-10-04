// Sample data - Replace this with code to load your actual results
const sampleData = {
  summary: {
    average_precision: 0.892,
    total_samples: 9562,
    confirmed_predictions: 2147,
    hz_candidates: 42,
    anomalies_detected: 763,
    f1_score: 0.874,
  },
  performance: {
    models: [
      "Logistic Regression",
      "Decision Tree",
      "Random Forest",
      "HistGB",
      "Ensemble",
    ],
    ap_scores: [0.72, 0.68, 0.89, 0.85, 0.91],
    f1_scores: [0.7, 0.65, 0.87, 0.83, 0.9],
  },
  hz_planets: {
    true: [
      "Kepler-186f",
      "Kepler-442b",
      "Kepler-62f",
      "Kepler-1229b",
      "Kepler-1652b",
      "Kepler-1544b",
      "Kepler-296e",
      "Kepler-1649c",
    ],
    predicted: [
      "Kepler-186f",
      "Kepler-442b",
      "Kepler-62f",
      "Kepler-1229b",
      "Kepler-1652b",
      "Kepler-1544b",
      "Kepler-296e",
      "Kepler-1649c",
      "Kepler-452b",
      "Kepler-1638b",
    ],
  },
  predictions: [
    {
      name: "Kepler-186f",
      prediction: "CONFIRMED",
      confidence: 0.95,
      hz: true,
      anomaly: false,
    },
    {
      name: "Kepler-442b",
      prediction: "CONFIRMED",
      confidence: 0.92,
      hz: true,
      anomaly: false,
    },
    {
      name: "KIC 10255705",
      prediction: "NOT_CONFIRMED",
      confidence: 0.15,
      hz: false,
      anomaly: false,
    },
    {
      name: "Kepler-1229b",
      prediction: "CONFIRMED",
      confidence: 0.88,
      hz: true,
      anomaly: false,
    },
    {
      name: "KIC 11026764",
      prediction: "NOT_CONFIRMED",
      confidence: 0.08,
      hz: false,
      anomaly: true,
    },
    {
      name: "Kepler-1652b",
      prediction: "CONFIRMED",
      confidence: 0.91,
      hz: true,
      anomaly: false,
    },
    {
      name: "Kepler-1544b",
      prediction: "CONFIRMED",
      confidence: 0.87,
      hz: true,
      anomaly: false,
    },
    {
      name: "KIC 11442793",
      prediction: "CONFIRMED",
      confidence: 0.76,
      hz: false,
      anomaly: false,
    },
  ],
};

// Function to load data from a JSON file
async function loadData() {
  try {
    // Try to load actual results from a JSON file
    const response = await fetch("/results.json");
    console.log(response);

    if (response.ok) {
      return await response.json();
    } else {
      console.log("No results.json found, using sample data");
      return sampleData;
    }
  } catch (error) {
    console.log("Error loading results, using sample data:", error);
    return sampleData;
  }
}

// Function to update the UI with data
function updateUI(data) {
  // Update summary metrics
  document.getElementById("apValue").textContent =
    data.summary.average_precision.toFixed(3);
  document.getElementById("totalSamples").textContent =
    data.summary.total_samples.toLocaleString();
  document.getElementById("confirmedValue").textContent =
    data.summary.confirmed_predictions.toLocaleString();
  document.getElementById("hzValue").textContent =
    data.summary.hz_candidates.toLocaleString();
  document.getElementById("anomaliesValue").textContent =
    data.summary.anomalies_detected.toLocaleString();
  document.getElementById("f1Value").textContent =
    data.summary.f1_score.toFixed(3);

  // Update last updated time
  document.getElementById(
    "lastUpdated"
  ).textContent = `Last updated: ${new Date().toLocaleDateString()}`;

  // Update HZ planet lists
  const trueHzContainer = document.getElementById("trueHzPlanets");
  const predictedHzContainer = document.getElementById("predictedHzPlanets");

  trueHzContainer.innerHTML = "";
  predictedHzContainer.innerHTML = "";

  data.hz_planets.true.forEach((planet) => {
    const planetItem = document.createElement("div");
    planetItem.className = "planet-item";
    planetItem.innerHTML = `<i class="fas fa-globe-americas"></i> ${planet}`;
    trueHzContainer.appendChild(planetItem);
  });

  data.hz_planets.predicted.forEach((planet) => {
    const planetItem = document.createElement("div");
    planetItem.className = "planet-item";
    planetItem.innerHTML = `<i class="fas fa-search"></i> ${planet}`;
    predictedHzContainer.appendChild(planetItem);
  });

  // Update results table
  const tableBody = document.getElementById("resultsTableBody");
  tableBody.innerHTML = "";

  data.predictions.forEach((pred) => {
    const row = document.createElement("tr");

    const statusClass =
      pred.prediction === "CONFIRMED"
        ? "status-confirmed"
        : "status-not-confirmed";
    const anomalyClass = pred.anomaly ? "status-anomaly" : "";
    const hzIcon = pred.hz
      ? '<i class="fas fa-sun" style="color: #f39c12;"></i>'
      : "";

    row.innerHTML = `
                    <td>${pred.name}</td>
                    <td><span class="status-badge ${statusClass}">${
      pred.prediction
    }</span></td>
                    <td>${(pred.confidence * 100).toFixed(1)}%</td>
                    <td>${hzIcon}</td>
                    <td>${
                      pred.anomaly
                        ? '<span class="status-badge status-anomaly">ANOMALY</span>'
                        : ""
                    }</td>
                `;

    tableBody.appendChild(row);
  });

  // Create performance chart
  createPerformanceChart(data.performance);
}

// Function to create performance chart
function createPerformanceChart(performanceData) {
  const ctx = document.getElementById("performanceChart").getContext("2d");

  new Chart(ctx, {
    type: "bar",
    data: {
      labels: performanceData.models,
      datasets: [
        {
          label: "Average Precision",
          data: performanceData.ap_scores,
          backgroundColor: "#3498db",
          borderColor: "#2980b9",
          borderWidth: 1,
        },
        {
          label: "F1 Score",
          data: performanceData.f1_scores,
          backgroundColor: "#9b59b6",
          borderColor: "#8e44ad",
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
          max: 1.0,
          title: {
            display: true,
            text: "Score",
          },
        },
      },
      plugins: {
        legend: {
          position: "top",
        },
        title: {
          display: true,
          text: "Model Performance Comparison",
        },
      },
    },
  });
}

// Initialize the page when loaded
window.addEventListener("DOMContentLoaded", async () => {
  const data = await loadData();
  updateUI(data);
});
